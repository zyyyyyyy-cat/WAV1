import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval
from torch.cuda import amp
import torch.distributed as dist
import torch.nn.functional as F
from loss.av1 import ModalityPrototypeBank, modality_prototype_loss, image_contrast_loss
from .av1_state import extract_centers, hard_matrix, manifest_hash, file_hash, model_hash, rng_state, restore_rng, load_trusted_artifact


def PCA(feat, target, num_modalities, group_size=4):
    """Progressive Center Alignment"""
    D = feat.shape[1]
    B = target.shape[0]

    feat_by_modality = feat.reshape(num_modalities, B, D)
    centers = []
    for m in range(num_modalities):
        fm = F.normalize(feat_by_modality[m], dim=1)
        cm = fm.reshape(-1, group_size, D).mean(1)
        cm = F.normalize(cm, dim=1)
        centers.append(cm)

    g_cent = F.normalize(torch.stack(centers, dim=0).mean(0), dim=1).detach()
    loss = sum((center - g_cent).pow(2).sum(1) for center in centers).mean() / num_modalities

    ids = target.reshape(-1, group_size)[:, 0].long()
    return loss, ids, g_cent


class PrototypeBank(nn.Module):
    def __init__(self, num_classes, feat_dim, momentum=0.2, device='cuda'):
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.m = momentum
        self.register_buffer('prototypes', torch.zeros(num_classes, feat_dim, device=device))

    @torch.no_grad()
    def update(self, ids_local, gcent_local, ddp=True):
        ids_local = ids_local.to(self.prototypes.device).contiguous()
        gcent_local = gcent_local.to(self.prototypes.device).contiguous()

        if ddp and dist.is_available() and dist.is_initialized():
            world = dist.get_world_size()
            ids_list = [torch.empty_like(ids_local) for _ in range(world)]
            gcent_list = [torch.empty_like(gcent_local) for _ in range(world)]
            dist.all_gather(ids_list, ids_local)
            dist.all_gather(gcent_list, gcent_local)
            ids_all = torch.cat(ids_list, dim=0)
            gc_all = torch.cat(gcent_list, dim=0)
        else:
            ids_all, gc_all = ids_local, gcent_local

        C, _ = self.prototypes.shape
        proto_sum = torch.zeros_like(self.prototypes)
        proto_count = torch.zeros(C, device=self.prototypes.device)

        proto_sum.index_add_(0, ids_all, gc_all)
        proto_count.index_add_(0, ids_all, torch.ones_like(ids_all, dtype=proto_count.dtype))

        mask = proto_count > 0
        new_centers = torch.zeros_like(self.prototypes)
        new_centers[mask] = proto_sum[mask] / proto_count[mask].unsqueeze(1)
        new_centers = F.normalize(new_centers, dim=1)

        old = self.prototypes[mask]
        upd = F.normalize((1 - self.m) * old + self.m * new_centers[mask], dim=1)
        self.prototypes[mask] = upd


def GPD(feat, target, bank: PrototypeBank, ids, g_cent, num_modalities, tau=0.03):
    """Global Prototype Discrimination"""
    target_rep = target.repeat(num_modalities)
    f = F.normalize(feat, dim=1)
    with torch.no_grad():
        P = bank.prototypes.clone()
        norms = P.norm(dim=1)
        need = norms[ids] < 1e-6
        if need.any():
            P[ids[need]] = g_cent[need]
        P = F.normalize(P, dim=1)

    logits = (f @ P.t()) / tau
    loss = F.cross_entropy(logits, target_rep)
    return loss


def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loaders,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_querys, local_rank, full_train_loaders=None):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = 'cuda'
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger('transreid.train')
    dist_on = (
        cfg.MODEL.DIST_TRAIN
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
    )
    rank = dist.get_rank() if dist_on else 0
    is_main = rank == 0
    if is_main:
        logger.info('start training')
    if device:
        model.to(local_rank)
        if dist_on:
            if is_main:
                logger.info('Using {} GPUs for DDP training'.format(dist.get_world_size()))
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank,
                # The ViT backbone retains its ImageNet classification head
                # (base.fc), while ReID training uses the outer classifier.
                # Consequently base.fc never participates in the loss.
                find_unused_parameters=True,
            )
            if local_rank == 0:
                torch.set_num_threads(16)
            else:
                torch.set_num_threads(4)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()
    meter_ls = [AverageMeter() for _ in range(4)]

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
    )
    scaler = amp.GradScaler()
    model_meta = model.module if hasattr(model, 'module') else model
    feat_dim = getattr(model_meta, 'in_planes', 768)
    num_classes = getattr(model_meta, 'num_classes', 500)
    loss_type = cfg.SOLVER.LOSS_TYPE.lower().strip()
    use_aux_modules = loss_type != 'base'
    if is_main:
        logger.info('Feature dimension: {}; identities: {}'.format(feat_dim, num_classes))
        logger.info('UAD start: seed=%d, ImageNet=%s, sha256=%s, batch=%d, epochs=%d',
                    cfg.SOLVER.SEED, cfg.MODEL.PRETRAIN_PATH,
                    file_hash(cfg.MODEL.PRETRAIN_PATH) if os.path.isfile(cfg.MODEL.PRETRAIN_PATH) else 'unavailable',
                    cfg.SOLVER.IMS_PER_BATCH, epochs)
    if use_aux_modules:
        proto_bank = PrototypeBank(
            num_classes=num_classes,
            feat_dim=feat_dim,
            momentum=cfg.MODEL.GPD_MOMENTUM,
            device=device,
        )

    av1 = cfg.AV1.ENABLED
    av1_sampler = (train_loader.batch_sampler.sampler if cfg.MODEL.DIST_TRAIN
                   else train_loader.sampler) if av1 else None
    if av1 and full_train_loaders is None:
        raise ValueError('A-V1 requires complete deterministic training loaders')
    mp_bank = ModalityPrototypeBank(num_classes, feat_dim, device=device) if av1 else None
    data_hash = manifest_hash(full_train_loaders) if av1 else None
    reference_path = os.path.join(cfg.OUTPUT_DIR, 'hard_reference_epoch010.pt')
    reference_hash = None
    start_epoch = 1
    skip_count = 0
    success_count = 0
    if cfg.AV1.RESUME:
        state = load_trusted_artifact(cfg.AV1.RESUME)
        if state['av1'] != av1 or (av1 and state['manifest_hash'] != data_hash):
            raise RuntimeError('Resume checkpoint does not match A-V1 mode or training manifest')
        model_meta.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        if use_aux_modules:
            proto_bank.load_state_dict(state['gpd_bank'])
        if av1:
            mp_bank.load_state_dict(state['mp_bank'])
            reference_path = state['reference_path']
            reference_hash = state['reference_hash']
        start_epoch = state['epoch'] + 1
        skip_count = state.get('skip_count', 0)
        success_count = state.get('success_count', 0)
        saved_rng = state['rng_by_rank'][rank]
        restore_rng(saved_rng)
        if start_epoch >= 11 and (not reference_path or not os.path.isfile(reference_path)
                                  or file_hash(reference_path) != reference_hash):
            raise RuntimeError('Missing or mismatched fixed epoch-10 hardness reference')
    elif av1:
        if is_main:
            centers, valid, _, _ = extract_centers(model_meta, full_train_loaders, num_classes, feat_dim, device)
            mp_bank.prototypes.copy_(centers)
            mp_bank.valid.copy_(valid)
            artifact = os.path.join(cfg.OUTPUT_DIR, 'mp_bootstrap.pt')
            torch.save({'bank': centers.cpu(), 'valid': valid.cpu(), 'manifest_hash': data_hash,
                        'pretrain_hash': file_hash(cfg.MODEL.PRETRAIN_PATH),
                        'modalities': list(cfg.DATASETS.MODALITIES),
                        'size_test': list(cfg.INPUT.SIZE_TEST)}, artifact)
            logger.info('MP bootstrap: %s sha256=%s', artifact, file_hash(artifact))
        if dist_on:
            dist.broadcast(mp_bank.prototypes, 0)
            dist.broadcast(mp_bank.valid, 0)
            dist.barrier()
    if av1 and start_epoch >= 11:
        ref = load_trusted_artifact(reference_path)
        if ref['manifest_hash'] != data_hash:
            raise RuntimeError('Hardness reference training manifest mismatch')
        av1_sampler.set_hardness(ref['hardness'].numpy())

    group_size = cfg.DATALOADER.NUM_INSTANCE

    for epoch in range(start_epoch, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        for m in meter_ls:
            m.reset()
        evaluator.reset()
        scheduler.step(epoch)
        model.train()
        if av1:
            av1_sampler.set_epoch(epoch)

        n_iter = 0
        for n_iter, (imgs, vid, camids, sample_ids) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()

            imgs = [img.to(device) for img in imgs]
            camids = [cam.to(device) for cam in camids]
            sample_ids = [ids.to(device) for ids in sample_ids]
            target = vid.to(device)

            num_modalities = len(imgs)
            target_rep = target.repeat(num_modalities)

            with amp.autocast(enabled=True):
                cls_score, global_feat, feat = model(imgs, target, camids)

                loss, il, tl = loss_fn(cls_score, global_feat, target_rep)
                meter_ls[0].update(il.item(), target_rep.shape[0])
                meter_ls[1].update(tl.item(), target_rep.shape[0])

                if use_aux_modules:
                    loss_pca, ids, g_cent = PCA(
                        global_feat,
                        target,
                        num_modalities=num_modalities,
                        group_size=group_size,
                    )
                    meter_ls[2].update(loss_pca.item(), max(1, target.shape[0] // group_size))

                    loss_gpd = GPD(
                        global_feat,
                        target,
                        proto_bank,
                        ids,
                        g_cent,
                        num_modalities=num_modalities,
                        tau=0.03,
                    )
                    meter_ls[3].update(loss_gpd.item(), max(1, target.shape[0] // group_size))

                    if 'pca' in loss_type:
                        loss += loss_pca * cfg.MODEL.PCA_LOSS_WEIGHT
                    if 'gpd' in loss_type:
                        loss += loss_gpd

                if av1:
                    with amp.autocast(enabled=False):
                        mp_loss, mp_counts = modality_prototype_loss(
                            global_feat, target, mp_bank, cfg.AV1.MP_TEMPERATURE)
                        img_loss, pair_loss, mixed_loss, img_counts = image_contrast_loss(
                            global_feat, target, camids, sample_ids, cfg.AV1.CONTRAST_TEMPERATURE)
                    ramp = min(epoch / 5.0, 1.0)
                    loss = loss + ramp * (cfg.AV1.MP_WEIGHT * mp_loss +
                                          cfg.AV1.CONTRAST_WEIGHT * img_loss)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            old_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            step_ok = scaler.get_scale() >= old_scale
            if dist_on:
                flags = torch.tensor([int(step_ok)], device=device)
                lower, upper = flags.clone(), flags.clone()
                dist.all_reduce(lower, op=dist.ReduceOp.MIN)
                dist.all_reduce(upper, op=dist.ReduceOp.MAX)
                if lower.item() != upper.item():
                    raise RuntimeError('GradScaler step success differs across ranks')
            success_count += int(step_ok)
            skip_count += int(not step_ok)

            if use_aux_modules and step_ok:
                with torch.no_grad():
                    proto_bank.update(ids, g_cent, ddp=cfg.MODEL.DIST_TRAIN)
                    if av1:
                        mp_bank.update(global_feat, target, distributed=dist_on)

            acc = (cls_score.max(1)[1] == target_rep).float().mean()
            loss_meter.update(loss.item(), target.shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize(local_rank)

            if is_main and (n_iter + 1) % log_period == 0:
                logger.info(
                    'Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}'.format(
                        epoch,
                        (n_iter + 1),
                        len(train_loader),
                        loss_meter.avg,
                        acc_meter.avg,
                        scheduler._get_lr(epoch)[0],
                    )
                )
                metrics = [f'{m.avg:.3f}' for m in meter_ls]
                metrics_str = ', '.join(metrics)
                logger.info(f'Epoch[{epoch}] {metrics_str}')
                if av1:
                    logger.info('A-V1 MP %.4f Pair %.4f Mixed %.4f; MP valid %s; image valid %s; AMP ok/skip %d/%d',
                                mp_loss.item(), pair_loss.item(), mixed_loss.item(),
                                mp_counts, img_counts, success_count, skip_count)

        end_time = time.time()
        if av1 and is_main:
            logger.info('A-V1 sampler epoch %d: %s; MP valid rows %d/%d; AMP ok/skip %d/%d',
                        epoch, av1_sampler.last_stats, int(mp_bank.valid.sum()), mp_bank.valid.numel(),
                        success_count, skip_count)
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        if dist_on:
            if is_main:
                logger.info(
                    'Epoch {} done. Time per batch: {:.3f}[s] Total: {:.1f}[s]'.format(
                        epoch, time_per_batch, end_time - start_time
                    )
                )
        else:
            logger.info(
                'Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'.format(
                    epoch, time_per_batch, train_loader.batch_size / time_per_batch
                )
            )

        if av1 and epoch == 10:
            if dist_on:
                dist.barrier()
            if is_main:
                epoch10_model_hash = model_hash(model_meta)
                centers, valid, counts, global_centers = extract_centers(
                    model_meta, full_train_loaders, num_classes, feat_dim, device)
                hardness = hard_matrix(centers, valid, counts, global_centers).cpu()
                torch.save({'centers': centers.cpu(), 'global_centers': global_centers.cpu(),
                            'valid': valid.cpu(),
                            'hardness': hardness, 'manifest_hash': data_hash,
                            'model_hash': epoch10_model_hash,
                            'modalities': list(cfg.DATASETS.MODALITIES),
                            'size_test': list(cfg.INPUT.SIZE_TEST),
                            'epoch': 10, 'temperature': 0.2}, reference_path)
                reference_hash = file_hash(reference_path)
                logger.info('Fixed hardness reference: %s sha256=%s', reference_path, reference_hash)
            if dist_on:
                dist.barrier()
            reference_hash = file_hash(reference_path)
            ref = load_trusted_artifact(reference_path)
            if ref['manifest_hash'] != data_hash:
                raise RuntimeError('Hardness reference manifest mismatch')
            av1_sampler.set_hardness(ref['hardness'].numpy())

        if epoch % checkpoint_period == 0 or (av1 and epoch == 10):
            local_rng = rng_state()
            if dist_on:
                all_rng = [None] * dist.get_world_size()
                dist.all_gather_object(all_rng, local_rng)
            else:
                all_rng = [local_rng]
            if is_main:
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(
                    model_to_save.state_dict(),
                    os.path.join(
                        cfg.OUTPUT_DIR, cfg.MODEL.NAME + '_{}.pth'.format(epoch)
                    ),
                )
                torch.save({'model': model_to_save.state_dict(), 'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                            'gpd_bank': proto_bank.state_dict() if use_aux_modules else None,
                            'mp_bank': mp_bank.state_dict() if av1 else None,
                            'epoch': epoch, 'av1': av1, 'manifest_hash': data_hash,
                            'reference_path': reference_path if av1 and epoch >= 10 else None,
                            'reference_hash': reference_hash if av1 and epoch >= 10 else None,
                            'rng_by_rank': all_rng, 'skip_count': skip_count,
                            'success_count': success_count},
                           os.path.join(cfg.OUTPUT_DIR, 'training_state_epoch{:03d}.pt'.format(epoch)))

        if epoch % eval_period == 0:
            eval_rng = rng_state()
            model.eval()

            world_sz = dist.get_world_size() if dist_on else 1
            if dist_on:
                # Validation assigns whole modalities to ranks, so ranks execute
                # different numbers of forward passes (and rank 3 is idle with
                # three modalities). Synchronize rank-0 buffers once and use the
                # unwrapped module to avoid DDP's per-forward buffer collective.
                for buffer in model.module.buffers():
                    dist.broadcast(buffer, src=0)
                dist.barrier()
                eval_model = model.module
            else:
                eval_model = model

            for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
                evaluator.set_query_num(mode, num_query)

                if dist_on and (mode - 1) % world_sz != rank:
                    continue

                for img, vid, camid, camidt, modid, _ in val_loader:
                    with torch.no_grad():
                        img = img.to(device)
                        feat = eval_model(img, mode=mode)
                        evaluator.update((feat, vid, camid, modid), mode)

            evaluator.split_all()
            cmc, mAP, *_ = evaluator.compute()

            if cmc is not None:
                logger.info(f'Validation Results - Epoch: {epoch}')
                logger.info(f'mAP: {mAP:.2%}')
                for r in (1, 5, 10):
                    logger.info(f'CMC curve, Rank-{r:<2}: {cmc[r-1]:.2%}')
            torch.cuda.empty_cache()
            restore_rng(eval_rng)


def do_inference(cfg,
                 model,
                 val_loaders,
                 num_querys):
    device = 'cuda'
    logger = logging.getLogger('transreid.test')
    logger.info('Enter inferencing')

    domain_report_path = None
    if cfg.TEST.DETAILED_DOMAIN_EVAL:
        domain_report_path = os.path.join(
            cfg.OUTPUT_DIR or ".", "domain_protocol_results.csv")

    evaluator = R1_mAP_eval(
        max_rank=50,
        feat_norm=cfg.TEST.FEAT_NORM,
        reranking=cfg.TEST.RE_RANKING,
        top_k=cfg.TEST.TOP_K_EVAL,
        logger=logger,
        detailed_domain_eval=cfg.TEST.DETAILED_DOMAIN_EVAL,
        domain_report_path=domain_report_path,
    )

    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    model.eval()
    img_path_list = []
    for mode, (val_loader, num_query) in enumerate(zip(val_loaders, num_querys), start=1):
        evaluator.set_query_num(mode, num_query)

        for img, vid, camid, camidt, modid, imgpath in val_loader:
            with torch.no_grad():
                img = img.to(device)

                feat = model(img, mode=mode)
                evaluator.update((feat, vid, camid, modid), mode)
                img_path_list.extend(imgpath)

    evaluator.split_all()
    cmc, mAP, *_ = evaluator.compute()

    logger.info('Validation Results ')
    logger.info('mAP: {:.2%}'.format(mAP))
    for r in [1, 5, 10]:
        logger.info('CMC curve, Rank-{:<3}:{:.2%}'.format(r, cmc[r - 1]))
    return cmc[0], cmc[4]
