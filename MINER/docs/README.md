# MINER paper website · WACV 2027

A standalone, responsive GitHub Pages microsite for **MINER: Modeling Inter- and Intra-Person Variance for Reidentification**.

## Publish with GitHub Pages

This folder belongs at `docs/` on the `main` branch of [oronnir/MINER](https://github.com/oronnir/MINER).

1. Open **Settings → Pages** in the GitHub repository.
2. Under **Build and deployment**, select **Deploy from a branch**.
3. Set **Branch: main**, **Folder: /docs**, then **Save**.
4. Wait for GitHub Pages to publish at **https://oronnir.github.io/MINER/**.

## Add arXiv and video links

Edit `docs/index.html` and find the constants near the bottom:

```js
const PAPER_URL = '';
const YOUTUBE_URL = '';
```

Replace the empty strings with the published arXiv abstract URL and the actual YouTube watch URL. The arXiv action will become a live button and the video will embed automatically.

The **Code on GitHub** button already points to the existing [MINER code repository](https://github.com/oronnir/MINER).

## Design notes

- Fully static HTML/CSS/JS; no build pipeline or JavaScript framework required.
- Responsive layouts and accessible diagrams.
- Method diagram and allocation charts are **explanatory illustrations** based on the MINER paper, not reproductions of its experimental plots.
- Abstract adapted from the uploaded preprint; the specific SOTA count has been softened pending up-to-date literature comparisons. Revisit the text when the arXiv abstract changes.
- Data-privacy friendly: no user analytics or tracking scripts. Google Fonts may request fonts externally (system-font fallback works offline).

## Preview locally

From this directory:

```bash
python -m http.server 8000
```

Open `http://localhost:8000`.
