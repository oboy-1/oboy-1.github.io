# Fourier Neural Operators Visually Explained

The files behind the post at `/posts/fourier-neural-operators/`. This folder isn't published as its own page. `_config.yml` excludes this README from the build.

## How it fits into the blog

Most posts here are markdown with small interactives embedded from `assets/files/<topic>/`. This post is an independent page instead: a scrolling explainer with two trained models that run in the browser, so it doesn't use the blog's post layout.

| Piece | Where |
|---|---|
| The page | `_posts/2026-10-20-fourier-neural-operators.html`. Its front matter sets `layout: none`, so Jekyll serves the HTML as written, at the normal `/posts/<slug>/` URL. Title, date, description and category come from the front matter, so the home card, archives, categories and search all work like any other post. |
| Everything it loads | This folder. The page links here with absolute paths. |
| Home-page card images | `assets/images/2026/10/fourier-sine-to-fluid-card-{light,dark}.jpg` |

`.gitignore` has a `!` exception for `vendor/` here, because the blog ignores folders named `vendor` everywhere else.

## Layout

```
css/style.css        the explainer's styles
css/site.css         the blog chrome around it: top bar, post date and category, footer
js/app.js            loads the data and models, runs the forward pass, drives the scroll
js/story.js          every slide in order: id, eyebrow, title, which data it shows, and its text
js/views/stage.js    picks the visual for the current slide (one `case` per slide id)
js/views/*.js        the visuals, grouped by topic:
  convolution.js       the convolution section (slides convwave … convwide)
  drill.js, learned.js, layer.js, buildup.js, flow.js, draw.js
  archscene.js         the 3D architecture scene; flowhero.js, the dye animation in the header
js/widgets/          the wave-sum and 1-D truncation widgets
js/fno.js            the model's forward pass in JavaScript
js/fourier.js        2-D FFT helpers; nn.js, small numeric helpers
js/data.js           loads a data folder (paths are relative to this file, not to the page)
js/render.js         painting fields, KaTeX helpers, and el() for building HTML
data/                the rock (Darcy) model: manifest.json, weights.wasm, test samples
data-ns/             the fluid (Navier-Stokes) model, same layout
vendor/              KaTeX, three.js, fft.js
```

## Common edits

- **Change a slide's text:** edit its entry in `js/story.js`.
- **Change a slide's visual:** find its id in `js/views/stage.js` to see which file draws it.
- **Add a slide:** add an entry to `STEPS` in `js/story.js` and a `case` for its id in `js/views/stage.js`. If it doesn't use the sample and grid picker, add its id to `NO_BAR` there too. Slide numbers are assigned by position, so nothing else needs renumbering.

The `.wasm` extension on the binary files is only so hosts serve them. They hold raw weights and samples, not WebAssembly.
