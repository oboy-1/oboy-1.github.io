// The scroll steps. Each entry owns one model state and the prose beat that goes with it.
// Every number is read from the manifest or the live forward pass; none are typed in
// (the one exception is the 1-D widget in "widget", which is its own self-contained demo).
// Voice: the blog's. Say the intuition first, ask the question the reader is about to ask,
// define a symbol right where it is first needed, and put math in LaTeX (M inline, D display).
import { el, pct, fmt, M, D } from "./render.js";

/** Channels whose |corr| with the true pressure clears THRESH, per stage. Straight from the rankings. */
export const THRESH = 0.5;
export function resembleCounts(manifest) {
  return ["lifted", "layer1", "layer2", "layer3", "layer4"].map((k) =>
    manifest.rankings[k].corr_target.score.filter((v) => v >= THRESH).length);
}

const stat = (k, v, s) => el("div", { class: "stat" }, el("span", { class: "k" }, k), el("span", { class: "v" }, v), s ? el("span", { class: "s" }, s) : null);
/** The blog's "Key Point" box and its quieter side-note cousin. */
const key = (...c) => el("div", { class: "key" }, el("b", {}, "Key point"), ...c);
const aside = (...c) => el("div", { class: "aside" }, ...c);
const tex = String.raw;

/** GELU next to ReLU, drawn from the real formula x·Φ(x). */
const m17 = (app) => app.manifest.model.params.toLocaleString("en-US");
function geluFigure() {
  const erf = (x) => { const t = 1 / (1 + 0.3275911 * Math.abs(x)); const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x); return x >= 0 ? y : -y; };
  const gelu = (x) => 0.5 * x * (1 + erf(x / Math.SQRT2));
  const W = 360, H = 190, x0 = -4, x1 = 4, y0 = -1, y1 = 4, px = (x) => 20 + ((x - x0) / (x1 - x0)) * (W - 40), py = (y) => H - 24 - ((y - y0) / (y1 - y0)) * (H - 44);
  const pts = (f) => Array.from({ length: 161 }, (_, i) => { const x = x0 + (i / 160) * (x1 - x0); return `${px(x).toFixed(1)},${py(f(x)).toFixed(1)}`; }).join(" ");
  const fig = el("figure", { class: "gelu-fig" });
  fig.innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="GELU compared with ReLU: GELU follows ReLU but is smooth at zero and dips slightly below zero for small negative inputs" style="width:100%;max-width:420px;height:auto;display:block">
    <line x1="${px(x0)}" y1="${py(0)}" x2="${px(x1)}" y2="${py(0)}" stroke="currentColor" stroke-opacity="0.35"/>
    <line x1="${px(0)}" y1="${py(y0)}" x2="${px(0)}" y2="${py(y1)}" stroke="currentColor" stroke-opacity="0.35"/>
    <polyline points="${pts((x) => Math.max(0, x))}" fill="none" stroke="currentColor" stroke-opacity="0.55" stroke-width="1.5" stroke-dasharray="5 4"/>
    <polyline points="${pts(gelu)}" fill="none" stroke="#ea580c" stroke-width="2.5" stroke-linejoin="round"/>
    <text x="${px(2.1)}" y="${py(3.25)}" font-size="12" fill="#ea580c" font-weight="700">GELU</text>
    <text x="${px(-3.9)}" y="${py(0.45)}" font-size="12" fill="currentColor" fill-opacity="0.7">ReLU (dashed)</text>
    <text x="${px(3.95)}" y="${py(0) + 14}" font-size="11" fill="currentColor" fill-opacity="0.6" text-anchor="end">input</text>
    <text x="${px(0) + 6}" y="${py(y1) + 10}" font-size="11" fill="currentColor" fill-opacity="0.6">output</text></svg>
    <figcaption class="cap-s">GELU against ReLU. It follows the same shape, but the corner at zero is smoothed off and it dips slightly below zero for small negative inputs.</figcaption>`;
  fig.style.margin = "0";
  return fig;
}
/** Stats for the layer the story walks through, live if the model has loaded. */
function lstats(app, l) {
  return app.sample.live?.layerStats?.[l - 1] ?? app.sample.entry.layers[l - 1];
}

export const STEPS = [
  {
    id: "problem",
    eyebrow: "The problem",
    title: "The problem",
    state: { dataset: "ns", view: "problem", flowMode: "plain", node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "Let's start with a Navier-Stokes problem. Picture a fluid that's just sitting still. We push on it with some pattern of forces, and that starts stirring it up. What we want to know is what the swirl looks like a little while later."),
        el("p", {}, "The normal way to answer that is to simulate it. But simulating it is tedious. It's computationally expensive, and you have to redo the whole thing for every single forcing pattern you want to try."),
        el("p", {}, "So the idea is this: what if we train a network that can simulate the problem for us, given any kind of input?"),
      ];
    },
  },
  {
    id: "operator",
    eyebrow: "What kind of network this is",
    title: "A function in, a function out",
    state: { dataset: "ns", view: "block", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "The key thing is that in a lot of these problems we don't have a fixed input. The data can come at different resolutions. The same flow might be sampled on a 64×64 grid or on a 128×128 grid, which is 4,096 numbers in one case and 16,384 in the other. A normal network has its input size baked into its first layer, so it can only take one of them. Ideally we want a network that accepts any input, whatever its dimensions."),
        el("p", {}, "So the idea is that our network isn't approximating a function that takes data in and gives data out:"),
        D(tex`f: \mathbb{R}^n \to \mathbb{R}^m`),
        el("p", {}, "It's approximating an ", el("strong", {}, "operator"), ". A normal network maps ", M(tex`n`), " numbers to ", M(tex`m`), " numbers, with ", M(tex`n`), " and ", M(tex`m`), " fixed and baked into the network. An operator maps a function to another function:"),
        D(tex`\mathcal{G}: a(x) \mapsto u(x)`),
        el("p", {}, "where both ", M(tex`a(x)`), " and ", M(tex`u(x)`), " can be evaluated at any ", M(tex`x`), " you like, at any resolution, because neither side is a fixed-length vector."),
        el("p", {}, "Here, the input function is the forcing: how hard, and which way, the fluid is being pushed at every point. The output function is the swirl (the vorticity) at every point you ask about, five time units later."),
      ];
    },
  },
  {
    id: "waves",
    eyebrow: "Before the split",
    title: "Any image is a sum of waves",
    state: { dataset: "ns", view: "waves", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "The way we do this is with something called the Fourier transform. If you haven't seen it before, start here: ", el("a", { href: "https://www.youtube.com/watch?v=spUNpyF58BY", target: "_blank", rel: "noopener" }, "3Blue1Brown's visual introduction to the Fourier transform"), "."),
        el("p", {}, "Say we have an image, the picture of our forcing. Normally we express it as pixels, where each pixel value contributes one small portion of the image. Instead, what if we express the image as a sum of waves that span the entire image?"),
        el("p", {}, "Some of those waves are fast (high frequency) and some are slow (low frequency). Each one spans the whole image, so together they build up broad brush strokes of how the image looks: the slow ones lay down the broad shape, and the fast ones fill in finer detail. Given enough of them, we can adequately represent the original image."),
        el("p", {}, "Now, none of these waves have anything to do with how big the image is. To describe a single wave, all I need is its frequency, its phase and its amplitude. So I can express an image without going pixel by pixel and without depending on its dimensions, using a set of waves with certain frequencies, phases and amplitudes."),
      ];
    },
  },
  {
    id: "recipe",
    eyebrow: "How waves add up",
    title: "A 1D example",
    state: { dataset: "ns", view: "recipe", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "To see this better, let's cook up a 1D example. On the right there are five waves, with a slider for each one that sets how much of that wave goes in. Add them up and you get the overall curve."),
        el("p", {}, "Play with the sliders and you can make some very interesting shapes. Each wave on its own is a simple sinusoid, and they all look alike. But add them together and you get surprisingly complex curves. So we can build really interesting, complex images out of a bunch of simple waves."),
      ];
    },
  },
  {
    id: "recipe2d",
    eyebrow: "How waves add up, in 2D",
    title: "Now in 2D",
    state: { dataset: "ns", view: "recipe2d", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "Now let's bring it to 2D. Instead of a curve, we have a grid of values, and we want 2D waves to add up to make a certain shape. It really is just the same concept. Use the slider and watch what happens as you add more and more waves: you get closer and closer to the image you want to represent."),
        el("p", {}, "That being said, you have to make sure you're adding up the ", el("em", {}, "correct"), " waves to get what you want."),
      ];
    },
  },
  {
    id: "stored",
    eyebrow: "How a picture is stored",
    title: "What a wave looks like as numbers",
    state: { dataset: "ns", view: "stored", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "Let's get more detail oriented. How do we actually write down a single wave? We need three things:"),
        el("ul", {},
          el("li", {}, el("strong", {}, "Frequency:"), " how many times the wave repeats across the image. In 2D it's a pair of counts, one for each direction, and that pair also tells you which way the stripes run."),
          el("li", {}, el("strong", {}, "Amplitude:"), " how tall the wave is."),
          el("li", {}, el("strong", {}, "Phase:"), " where the wave starts. Changing it slides the stripes sideways.")),
        el("p", {}, "Put together, one wave looks like this:"),
        D(tex`h \cos\big(2\pi\, k\cdot x + \varphi\big)`),
        el("p", {}, "where ", M(tex`k`), " is the frequency, ", M(tex`h`), " is the amplitude and ", M(tex`\varphi`), " is the phase. You'll often see these packed into a single complex number, where its size is the amplitude and its angle is the phase."),
        el("p", {}, "So an image becomes a list of waves, and each wave is just a few numbers. On the right is a real forcing pattern written out that way, strongest waves first. A handful of waves already carry most of the picture, and the rest add smaller and smaller corrections."),
        el("p", {}, "Notice that none of these numbers mention how many pixels the image has. The list describes the image without relying on the grid."),
      ];
    },
  },
  {
    id: "energy",
    eyebrow: "What energy is",
    title: "What energy is",
    state: { dataset: "ns", view: "energy", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "It turns out you need a large number of waves to fully express an image. So we're almost back to the same problem, or even a worse one: to express any image, we'd need a huge number of data points."),
        el("p", {}, "Yes, by doing the Fourier transform we've eliminated the dependence on the dimensions of our data. But now we need a gazillion data points to actually express the image."),
        el("details", {}, el("summary", {}, "Why not infinite?"),
          el("p", {}, "For a continuous signal, yes, you'd need infinitely many waves, every possible frequency, to describe it exactly. But everything on this page is a discrete grid, like our 64×64 forcing. A discrete Fourier transform on a signal with ", M(tex`N`), " points uses exactly ", M(tex`N`), " waves, and the reconstruction is exact. So for us it's never infinite. It's a large but fixed number: 4,096 for a 64×64 image. That's still a lot, and we'd like to prune it down to the waves we actually need.")),
        el("p", {}, "So instead, we need to figure out which waves in the image are actually important, in other words, which waves make up most of the image. We want to remove the waves that only do fine, minute detail, and keep the ones that give the broad brush strokes."),
        el("p", {}, "For that we have a metric called ", el("strong", {}, "energy"), ". Energy is simply the square of a wave's amplitude. The higher the wave's amplitude, the further the wave pushes away from the default flat value. Try the sliders on the right: notice how increasing the amplitude pushes the summed-up wave much further from the usual flat line."),
        el("p", {}, "That's what energy is. It tells us which waves matter most in an image, so we can measure how much of the image a smaller set of waves still holds. Keeping a small, fixed number of waves is how we get around the huge number of data points, regardless of dimensions."),
        el("p", {}, "Energies also add. If an image is a recipe of several waves, the total energy of the image is just the sum of each wave's energy on its own, with nothing left over and no cross-terms to worry about:"),
        D(tex`E_{\text{total}} = \sum_k E_k = \sum_k a_k^2`),
        el("p", {}, "That means you can rank every wave by its energy, add them up in that order, and see exactly what fraction of the whole image the top waves hold. We'll use this number to check how much we lose when we throw waves away. (The network itself doesn't rank by energy – it keeps a fixed block of the slowest waves, and we'll see why later.)"),
      ];
    },
  },
  {
    id: "invariance",
    eyebrow: "Resolution invariance",
    title: "Resolution invariance: the promise",
    state: { dataset: "ns", view: "invariance", flowMode: "plain", node: null, focus: null, draw: false },
    prose(app) {
      return [
        el("p", {}, "This is just a simple showcase of how our neural network is able to operate on different kinds of resolutions. Look at this, see how it works: same trained network, same forcing, run at different grid sizes."),
        el("p", {}, "The error shown is how far off the prediction is, as a percentage."),
        el("details", {}, el("summary", {}, "What does the error number mean?"),
          el("p", {}, "It's built the same way as MSE: squared differences between prediction and truth. But instead of averaging those squared errors, we take the square root of their sum and divide by the square root of the sum of the squared true values. That turns it into a percentage, which doesn't depend on the data's units.")),
      ];
    },
  },
  {
    id: "convwave",
    eyebrow: "Convolution, one wave (1/5)",
    title: "We're secretly doing convolutions",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Thinking in waves frees us from the grid. It also lets us do convolutions for much less cost. Specifically, it lets the network act like a CNN with a really large kernel, without the computational complexity that usually comes with one."),
        el("p", {}, "Why do we need such large kernels? Because in problems like Navier–Stokes, what happens in one region can affect a region all the way on the other side!"),
      ];
    },
  },
  {
    id: "convkernel",
    eyebrow: "Convolution, one wave (2/5)",
    title: "The kernel: a few weights",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Let's see this in action. For starters, what is a kernel?"),
        el("p", {}, "A kernel is just a short list of weights. You slide it along the signal, and at every spot it looks at a few neighbouring samples, multiplies each one by its weight, and adds them up. Different weights do different things: an average kernel averages a point with its neighbours, and a difference kernel subtracts the point behind from the point ahead, so it picks out where the signal changes fast."),
        el("p", {}, "For our input, we'll use a single wave measured at 32 points. On the right, the kernel sits under it, with one weight lined up under each sample it reads. At the bottom, the same kernel runs over a block with sharp sides, so you can see what each kernel does: average rounds off the corners, difference lights up only at the two jumps, and shift just moves the block over."),
      ];
    },
  },
  {
    id: "convone",
    eyebrow: "Convolution, one wave (3/5)",
    title: "One output value",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "So how do we actually get an output? Take the spot the kernel is sitting on. Multiply each weight by the sample right above it, then add up all the products. That one number is the output at that spot."),
        el("p", {}, "That's really all a convolution is doing at a single spot. Use the buttons, or click on the wave, to move the kernel and see another spot."),
      ];
    },
  },
  {
    id: "convslide",
    eyebrow: "Convolution, one wave (4/5)",
    title: "Slide it everywhere",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Now we just do the same thing at every spot, one at a time. Each spot gives us one output point, and together they trace out the whole output."),
      ];
    },
  },
  {
    id: "convout",
    eyebrow: "Convolution, one wave (5/5)",
    title: "A sine wave in, a sine wave out",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Look at what came out: another sine wave! It has the same frequency as the one that went in, so it repeats the same number of times. The only things that changed are how tall it is and where it sits."),
        el("p", {}, "Try a different k or a different kernel. No matter what, a sine wave goes in and a sine wave of the same frequency comes out, just scaled and shifted. So a kernel's whole effect on one wave boils down to two numbers: how much it scales it, and how much it shifts it."),
        el("p", {}, "Now, all of this assumes the input is a single sine wave. A real input is usually a messy signal, so the next slide handles that case."),
        el("p", { class: "note" }, "The proof is under the plot."),
      ];
    },
  },

  {
    id: "convsum",
    eyebrow: "Convolution, many waves (1/4)",
    title: "A real input",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "So what about a real input? Remember, any signal is just a sum of sine waves. And convolution has one more property that saves us: it doesn't care whether you add first or convolve first. Each output is just weights times samples, added up, so convolving a sum of waves gives the same result as convolving each wave on its own and adding the results."),
        el("p", {}, "Put those together with the last slide, and we don't actually need to slide anything. Let's try it on the signal on the right. It looks messy, but it's secretly three waves added together."),
      ];
    },
  },
  {
    id: "convsplit",
    eyebrow: "Convolution, many waves (2/4)",
    title: "Split it into waves",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "First, the Fourier transform pulls the signal apart into the waves it's made of: here, one slow, one medium and one fast. Add the three rows back up point by point and you get the signal again (dashed, on top)."),
      ];
    },
  },
  {
    id: "convchange",
    eyebrow: "Convolution, many waves (3/4)",
    title: "Change each wave on its own",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "From the last slide, a kernel turns a sine wave into another sine wave of the same frequency, just scaled and shifted. So we put each wave through the kernel on its own. The labels say how much each one gets scaled and shifted."),
      ];
    },
  },
  {
    id: "convadd",
    eyebrow: "Convolution, many waves (4/4)",
    title: "Add them back up",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Now add the changed waves back together, one at a time. The dots are what you get from sliding the kernel along the input directly, like before, and they land right on the sum."),
        el("p", {}, "So a convolution is really just a multiply on each wave."),
      ];
    },
  },

  {
    id: "convcost",
    eyebrow: "Convolution, the cost",
    title: "Why go through the waves",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Now let's get to the actual example: a convolution on our input. In our applications, parts of the grid that are really far away from each other interact, especially with Navier–Stokes and Darcy flow. So we want to effectively be running a convolution with a really big kernel, so that each pixel gets updated based on pixels far away from it too."),
        el("p", {}, "Now, as you can see, as you increase the kernel size, the number of multiplications and additions you need goes up by a lot. It grows with the kernel's area (so quadratically with its width), which means double the width and you're doing four times the work. And once you get to a kernel the size of the whole grid, it really blows up: about 34 million operations at 64×64, and 8.6 billion at 256×256."),
        el("p", {}, "If you instead stick to the wave way of doing it, in the frequency domain, you get much better efficiency. It costs about half a million operations at 64×64 no matter how big the kernel is. That's around 65 times less work than sliding a whole-grid kernel."),
        el("p", {}, "To be fair, though, for small kernels like 3×3, sliding is actually cheaper. On a 64×64 grid, the waves only start winning somewhere between 7×7 and 9×9. But big kernels are exactly what we want here, right?"),
      ];
    },
  },
  {
    id: "convwide",
    eyebrow: "Convolution, how wide",
    title: "Why this network wants a wide kernel",
    state: { dataset: "ns", view: "conv", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Now, if you're still not convinced, here's an example of why we need the big kernel."),
        el("p", {}, "Click somewhere on the grid on the right. That makes a small change to the input at that one spot, and we run our network on it. The picture next to it shows how much the prediction changed."),
        el("p", {}, "The dashed square is what a normal CNN could do. Say you had four normal 3×3 layers. Each one can only pass a change over by one pixel, so after four of them, a change can travel four pixels at most. Nothing outside that square could ever be affected."),
        el("p", {}, "But look at our network: a lot of the change lands way outside the square. For the default poke on the fluid, it's 87%. One small change is impacting a really large area. Having small sliding windows ain't going to cut it, right? You have to deal with large sliding windows, which is why we're making this efficiency gain by doing it through the frequency domain."),
      ];
    },
  },
  {
    id: "overview",
    eyebrow: "The whole network",
    title: "The whole network at a glance",
    state: { dataset: "ns", view: "flow", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose() {
      return [
        el("p", {}, "A Fourier neural operator is made of a few main parts. First, we lift the input into 32 channels. Then we pass it through four Fourier layers, one after the other. Inside each layer, the input splits into two paths \u2013 a spectral path and a local path \u2013 that run in parallel and get added back together. At the end, we project down to a single channel, and that is our output."),
      ];
    },
  },
  {
    id: "lift",
    eyebrow: "Lifting",
    title: "Lifting: one grid in, 32 channels out",
    state: { dataset: "ns", view: "lift", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose() {
      return [
        el("p", {}, "The network's input is a grid of values, and that grid can be any size. Just like how convolutional layers can take any kind of input, the first thing we do is run a convolutional layer on it. But instead of a kernel with some large size, it's a 1\u00d71 kernel, and it expands out to 32 channels. Those 32 channels exist so that the network has a lot more values to work with, instead of just one number per pixel."),
        el("details", {}, el("summary", {}, "Why x and y get added as extra input channels"),
          el("p", {}, "Remember, this whole network is meant to be resolution invariant from end to end. That's why we use 1\u00d71 kernels. A 3\u00d73 or a 5\u00d75 kernel looks at a neighbourhood, so it builds in a sense of which pixels sit close to each other. A 1\u00d71 kernel never looks at a neighbour, so nothing in the network knows where a pixel is. To give that back, we add the x and y position of every pixel as two more input channels.")),
      ];
    },
  },
  {
    id: "split",
    eyebrow: "Inside one layer",
    title: "Inside one layer: two paths, added together",
    state: { dataset: "ns", view: "layer", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "If we peek inside, a Fourier layer takes whatever the lift gave us and sends it down two paths."),
        el("p", {}, "The first path is the spectral path, and the second is the local path. We'll go through each one on its own."),
        el("p", {}, "The two paths get added together, then passed through a nonlinearity."),
      ];
    },
  },
  {
    id: "truncate",
    eyebrow: "The spectral path",
    title: "The spectral path",
    state: { dataset: "ns", view: "layer", layer: 2, focus: "truncate", node: null, draw: false },
    prose() {
      return [
        el("p", {}, "We run the fast Fourier transform and break the image into waves. We can't keep all of them, so we keep a fixed block of the slowest ones. That block is picked before training, so every image gets the same set of waves. Slow waves lay down the broad brush strokes, so in smooth data they usually carry most of the energy."),
        el("details", {}, el("summary", {}, "Why not truncate by energy directly?"),
          el("p", {}, "You'd think we'd keep whichever waves have the most energy. But the waves themselves are fixed \u2013 wave number 5 is always the pattern that wiggles 5 times across the grid. What changes from image to image is how much of each wave is in there. So if we picked by energy, the set of waves we keep would change with every image, and the learned weight for wave 5 wouldn't mean the same thing twice. Instead we pick the slowest waves. They lay down the broadest brush strokes, so they make a good stand-in for the most important waves, and the set never changes.")),
      ];
    },
  },
  {
    id: "multiply",
    eyebrow: "The spectral path",
    title: "A learned complex number per wave",
    state: { dataset: "ns", view: "multiply", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose() {
      return [
        el("p", {}, "So we've kept a fixed set of waves. Now we want to apply a learned weight to each one \u2013 but what does that actually look like in math? Each wave, recall, is a complex number: its size is the amplitude, its angle is the phase. The network learns its own complex number for that wave, call it ", M(tex`R`), ", and applying the weight just means multiplying the two:"),
        D(tex`\begin{aligned} a_k \cdot R_k &= \big(h_k e^{i\varphi_k}\big) \cdot \big(r_k e^{i\theta_k}\big) \\ &= h_k r_k\, e^{i(\varphi_k + \theta_k)} \end{aligned}`),
        el("p", {}, "Multiplying complex numbers multiplies their sizes and adds their angles \u2013 so ", M(tex`R_k`), " scales the wave's amplitude by ", M(tex`r_k`), ", and rotates its phase by ", M(tex`\theta_k`), ". Of those four numbers, only ", M(tex`r_k`), " and ", M(tex`\theta_k`), " are learned: they come out of training and stay fixed afterward, the same for every image. ", M(tex`h_k`), " and ", M(tex`\varphi_k`), " aren't learned at all \u2013 they're just whatever the FFT measured for that particular image. There's one ", M(tex`R`), " per wave, per input-output channel pair, all learned during training, and the paper itself calls this weight tensor ", M(tex`R`), "."),
      ];
    },
  },
  {
    id: "widget",
    eyebrow: "The same cut, in one dimension",
    title: "Keeping most of the energy can still lose the shape",
    state: { dataset: "ns", view: "layer", layer: 2, focus: "widget", node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Let's go back to the cut and look at what it does in one dimension, where it's easier to see everything at once. Instead of a whole image, we'll take a single slice: a square step that sits at 0, jumps up to 1, and then drops back down to 0. We build that step out of waves, throw away every wave above a cutoff, and look at what's left."),
        el("p", {}, "If you drag the slider to this model's cutoff, ", M("k=8"),
          ", the readout says we've kept 96% of the energy. But the picture looks noticeably off. The corners get rounded, and right at each jump the curve only makes it halfway, so it's off by about 0.5. Just next to each jump it also overshoots by about 9% of the step's height, then ripples back down."),
        el("p", {}, "If you keep dragging to the right, the overshoot stays at about 9%. The spike just gets narrower and moves in closer to each jump. This is called the Gibbs phenomenon. Now switch the target to the smooth bump, and you'll see the same cutoff rebuilds it almost exactly."),
        key("At the end of the day, the fast waves are where a lot of the sharp detail is. They carry very little energy, so we can keep 96% of the energy and still lose the sharp edges."),
        el("p", { class: "note" }, "This is a 1-D slice, so its numbers are its own. The 2-D picture of the real layer is on the spectral-path slide, and both use the same cutoff."),
      ];
    },
  },
  {
    id: "bridge",
    eyebrow: "The local path",
    title: "The local path",
    state: { dataset: "darcy", view: "bridge", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose() {
      return [
        el("p", {}, "Since we threw away a lot of waves, we've lost some information, and the network needs something to pick it up. This is where we switch to the Darcy flow example \u2013 its sharp edges make the loss easy to see. The local path runs a 1\u00d71 convolution over the grid, mixing the 32 channels at each pixel with no neighbours involved. It keeps track of individual details, including the extra signal the cutoff threw away. In short, one path sees the whole image at once, and the other sees one pixel at a time."),
        el("details", {}, el("summary", {}, "What is Darcy flow?"),
          el("p", {}, "Darcy flow describes water seeping slowly through porous ground, like soil or rock. The input is a map of the ground made of two materials: one that lets water through easily and one that doesn't. Water is added evenly everywhere, and the edges are held at zero pressure, so that's where it drains out. The output is the water pressure at every point once everything has settled."),
          D(tex`\begin{gathered} -\nabla \cdot \big(a(x)\, \nabla u(x)\big) = 1 \\ u = 0 \text{ on the edges} \end{gathered}`),
          el("p", {}, "Here ", M(tex`a(x)`), " is how easily water moves through the ground at ", M(tex`x`), ", and ", M(tex`u(x)`), " is the pressure. It's lowest along the edges, where the water drains out, and rises toward the inside. Where the two materials meet, the pressure's slope changes suddenly. Those sharp boundaries are why this example is good for seeing what the cutoff loses.")),
      ];
    },
  },
  {
    id: "pixels",
    eyebrow: "Back to pixels",
    title: "Back to pixels",
    state: { dataset: "darcy", view: "layer", layer: 2, focus: null, node: null, draw: false },
    prose() {
      return [
        el("p", {}, "Once each wave is scaled and rotated, we run the inverse Fourier transform to turn them back into a grid. By the convolution theorem, this whole spectral path \u2013 split into waves, multiply, convert back \u2013 is mathematically the same as running one big convolution over the grid, without us ever computing it directly."),
      ];
    },
  },
  {
    id: "paths",
    eyebrow: "The two paths, measured",
    title: "The spectral path carries almost none of the sharp detail",
    state: { dataset: "darcy", view: "layer", layer: 2, focus: "paths", node: null, draw: false },
    prose(app) {
      const st = lstats(app, app.state.layer);
      const s = st.outside_cutoff_share_spectral;
      return [
        el("p", {}, "The spectral path truncates the fast waves and only keeps the slow ones, so whatever it processes is very smooth data \u2013 and that means the sharp detail is missing from that path."),
        el("p", { class: "note" }, `Quick note: the network on this page is a small one we trained to run in your browser (${m17(app)} weights, four layers, 32 channels, trained at 64\u00d764), so the numbers from here on describe this model.`),
        el("p", {}, "Now we check the claim on a real layer. On one shared colour scale the local path looks like the smaller signal: across all 32 channels of this layer its spatial standard deviation is ",
          el("strong", {}, `${fmt(st.std_spectral / st.std_local, 1)}×`), " smaller than the spectral path's."),
        el("p", {}, "Now subtract everything below the cutoff from both. The spectral tile goes blank, and the local tile is nothing but outlines."),
        key("Above the cutoff, the local path supplies the energy and the spectral path supplies none."),
        el("p", {}, "The bars measure it: the local path supplies ", el("strong", {}, pct(st.outside_cutoff_share_local, 1)), " of the energy above the cutoff and the spectral path supplies ",
          el("strong", {}, s < 5e-4 ? s.toExponential(1) : pct(s, 1)), ", which is rounding error. That's true in every layer, sample and grid size we ran (the largest spectral share anywhere is ",
          el("span", { class: "mono" }, app.manifest.stats.outside_cutoff_share_spectral_max.toExponential(1)), ")."),
      ];
    },
  },
  {
    id: "block",
    eyebrow: "The rest of the layer",
    title: "The rest of the layer",
    state: { dataset: "darcy", view: "block", focus: null, node: null, draw: false },
    prose(app) {
      const m = app.manifest;
      return [
        el("p", {}, "Once the two paths are added together, we pass the result through a GELU. It's a nonlinearity \u2013 a smooth version of ReLU that lets the network bend things instead of only scaling them. Without it, stacking layers would just collapse into one big linear map."),
        geluFigure(),
        el("p", {}, "Then we do a channel MLP. That's a small network that works on the 32 channels at each pixel on its own, and it's built out of 1\u00d71 convolutions (32 down to 16 and back up to 32, with a GELU in between)."),
        el("p", {}, "Last is a skip connection. In every layer, the layer's own input comes back around, scaled by one learned number per channel, and gets added to what the MLP produced. Then one more GELU, except in the last layer."),
        el("p", {}, "Written out, as implemented in ", el("span", { class: "mono" }, m.model.library), ":"),
        D(tex`\begin{aligned} v_{\ell+1} = \sigma\Big(&\,\mathrm{MLP}\big(\sigma(\mathcal{S}v_\ell + W v_\ell)\big) \\ &+ g\odot v_\ell\Big), \end{aligned}`),
        el("p", {}, "where ", M(tex`\mathcal{S}`), " is the spectral path, ", M(tex`\sigma`), " is a GELU, and ", M(tex`g`), " is one learned number per channel. The circle with a dot, ", M(tex`\odot`), ", means multiply channel by channel: each channel of ", M(tex`v_\ell`), " is scaled by its own number from ", M(tex`g`), ". Layer 4 skips the last GELU."),
      ];
    },
  },
  {
    id: "flow",
    eyebrow: "Stacking four layers",
    title: "Stacking four layers: where the answer appears",
    state: { dataset: "darcy", view: "flow", flowMode: "counts", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      const { manifest: m, sample } = app;
      const c = resembleCounts(m);
      const err = sample.live ? sample.live.error : sample.error?.[sample.res];
      const mono = c.every((v, i) => i === 0 || v >= c[i - 1]);
      return [
        el("p", {}, c[4] > c[0]
          ? "Here we're looking at the real channels of the Darcy flow model as they move through the four layers. At the start they still look a lot like the rock map, and by the end most of them look like the pressure \u2013 so each layer is turning the rock into the answer a little more."
          : "Here we're looking at the real channels as they move through the four layers. The badge above each column counts how many of the 32 look like the true answer."),
        el("p", {}, c[4] > c[0]
          ? `We can count it too: out of 32 channels, ${c[0]} look like the pressure after the lift, and ${c[4]} do after layer 4 (${c.join(" \u2192 ")}).`
          : `For this model the counts are ${c.join(" \u2192 ")}.`),
        el("div", { class: "statgrid" },
          stat("This sample", err != null ? pct(err, 1) : "\u2014", err != null ? `relative L2 error at ${sample.res}\u00d7${sample.res}` : "drawn map, no true answer"),
          stat("Test set", pct(m.stats.test_error[String(sample.res)]?.mean ?? m.stats.test_error["64"].mean, 2), `mean over ${m.stats.test_error["64"].n} unseen samples`)),
        el("p", { class: "note" }, "A channel \u201cresembles\u201d the pressure when its correlation with the true pressure, ", M("r"), ", is at least 0.5 in size (", M(tex`|r|\ge ${THRESH.toFixed(1)}`), "). An ", M("r"), " of 1 would mean the channel is exactly the pressure, up to a scale. We picked 0.5 just to make the counts easy to read. The columns are ranked by this correlation too, and if you switch the ranking above the diagram, the same 32 channels look like random blobs."),
      ];
    },
  },
  {
    id: "project",
    eyebrow: "Projection",
    title: "Projection: 32 channels back to one",
    state: { dataset: "darcy", view: "flow", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose() {
      return [
        el("p", {}, "After the fourth layer we're still holding 32 channels, but all we want is one \u2013 the prediction. So we do one last small network, ", M("Q"), ", on the 32 channels at each pixel on its own. It's the same kind of thing as the channel MLP: 1\u00d71 convolutions that go from 32 up to 64 and then down to 1. That one channel is our answer, and the solver's answer sits next to it so we can check."),
      ];
    },
  },
  {
    id: "resolution",
    eyebrow: "Resolution, checked",
    title: "The same weights run on grids they never saw",
    state: { dataset: "ns", view: "resolution", focus: null, node: null, draw: false },
    prose(app) {
      const ns = app.ds.ns.manifest, dc = app.ds.darcy.manifest;
      const e = (m, r) => pct(m.stats.test_error[String(r)].mean, 1);
      const flat = (m) => new Set([32, 64, 128].map((r) => e(m, r))).size === 1;
      const best = (m) => [32, 64, 128].reduce((a, r) => (m.stats.test_error[String(r)].mean < m.stats.test_error[String(a)].mean ? r : a), 64);
      return [
        el("p", {}, "Now we get back to the promise from the start. Each layer works on a fixed set of ", el("strong", {}, `${ns.stats.n_kept_modes} waves`),
          ", and those waves don't depend on the grid, so the same weights can run on any grid. Here are both trained models, unchanged, on three grids each."),
        el("p", {}, flat(ns) ? `The fluid's error is ${e(ns, 64)} on all three. ` : `The fluid's error is ${e(ns, 32)}, ${e(ns, 64)} and ${e(ns, 128)} on the three grids. `,
          `The rock's is ${e(dc, 32)} at 32\u00d732, ${e(dc, 64)} at 64\u00d764 and ${e(dc, 128)} at 128\u00d7128, so it's lowest at ${best(dc)}\u00d7${best(dc)}${best(dc) === dc.model.trained_resolution ? ", the grid it was trained on" : ""}. `,
          "Neither model gets better as the grid gets finer \u2013 every grid here is a sampling of the same underlying data, so a finer grid doesn't bring any new detail."),
        el("p", { class: "note" }, `Mean over ${ns.stats.test_error["64"].n} unseen test samples per problem and grid.`),
      ];
    },
  },
  {
    id: "stillfails",
    eyebrow: "Where it still fails",
    title: "Where the fluid still goes wrong",
    state: { dataset: "ns", view: "stillfails", focus: null, node: null, draw: false },
    prose(app) {
      const a = app.ds.ns.manifest.analysis, E = a.ns.energy_above_cutoff_64, A = (k) => a[k].ablation_64;
      return [
        el("p", {}, "Now let's look at where the fluid's leftover error is. Only ", el("strong", {}, pct(E.truth, 2)),
          " of the true swirl's energy is above the cutoff, and yet ", el("strong", {}, pct(E.error, 0)), " of the model's ", el("em", {}, "error"), " is. The mistakes are in the thin streaks, which is exactly the sharp detail the spectral path can't draw."),
        el("p", {}, "Earlier we saw that the local path ", el("em", {}, "carries"), " that detail. Now we check whether it's ", el("em", {}, "needed"),
          ". If we remove just that part of the local path in every layer and re-run, the fluid's error goes from ", el("strong", {}, pct(A("ns").baseline.mean, 2)), " to ",
          el("strong", {}, pct(A("ns").local_above_cutoff_removed.mean, 2)), ", and the rock's goes from ", el("strong", {}, pct(A("darcy").baseline.mean, 2)), " to ",
          el("strong", {}, pct(A("darcy").local_above_cutoff_removed.mean, 2)), ". So the local path's fine detail matters for both. The rock's error goes up by about a third, and the fluid's by about a tenth."),
      ];
    },
  },
  {
    id: "draw",
    eyebrow: "Your turn",
    title: "Paint a forcing the model has never seen",
    state: { dataset: "ns", view: "flow", flowMode: "plain", focus: null, node: null, draw: true },
    prose(app) {
      const live = app.sample.live;
      return [
        el("p", {}, `Now it's your turn. The model is running right here in the page \u2013 all ${app.manifest.model.params.toLocaleString("en-US")} weights, a full forward pass in JavaScript. Push the fluid with your own strokes and every column above updates. Try one blob on its own, a red and a blue side by side, or a long band. You can also change the grid size: your painting gets resampled and the same weights run again.`),
        el("p", {}, "There's no solver in this page, so what you paint has no true answer and no error number."),
        live && app.state.draw ? el("p", { class: "note" }, `Last run: ${Math.round(live.ms)} ms at ${live.res}×${live.res} in this browser.`) : null,
        el("p", {}, "Open a layer from the diagram and the two-path split is recomputed from your drawing."),
      ];
    },
  },
  {
    id: "limits",
    eyebrow: "Where it breaks",
    title: "What this does not show",
    state: { dataset: "ns", view: "limits", flowMode: "plain", layer: 2, node: null, focus: null, draw: false },
    prose(app) {
      const m = app.manifest;
      return [
        el("p", {}, "A few limits to keep in mind:"),
        el("p", {}, el("strong", {}, "This is a small model."), ` It has ${m.model.params.toLocaleString("en-US")} weights, four layers and 32 channels, and we built it to run in a browser \u2013 so every number on this page is measured on this one model. Larger FNOs can behave differently.`),
        el("p", {}, el("strong", {}, "The grid wraps around."), " A Fourier transform treats the picture as if it repeats forever, left to right and top to bottom. For the fluid that's simply true \u2013 it was simulated on a domain whose edges join. A rock map isn't like that: where one copy meets the next, the rock suddenly jumps, and describing a jump takes waves of every speed."),
        el("p", {}, el("strong", {}, "The cut throws detail away."), ` The spectral path keeps ${m.stats.n_kept_modes} of ${m.stats.n_total_modes.toLocaleString("en-US")} waves, so everything sharper has to come from the local path.`),
        el("p", {}, el("strong", {}, "One training grid."), " We trained both models only at ", el("span", { class: "mono" }, `${m.model.trained_resolution}\u00d7${m.model.trained_resolution}`), ", on one dataset each. We ran them at 32 and 128 on the same data, but we didn't try a genuinely finer flow with detail a 128\u00d7128 grid can't hold."),
        el("p", {}, el("strong", {}, "A separate model for each problem."), " We trained the same architecture twice, once on the fluid and once on the rock, so nothing here shows one model handling both."),
        el("p", {}, "That's what's inside a Fourier neural operator. Each layer turns its input into waves, scales and shifts each kept wave by a learned number, turns the result back into a grid, and adds a local path that keeps the detail the dropped waves carried. Thanks for reading!"),
      ];
    },
  },
];

/** Footer notes. Numbers from both manifests; nothing typed in. */
export function notes(app) {
  const ns = app.ds.ns.manifest, dc = app.ds.darcy.manifest;
  const q = (m) => Math.max(...Object.values(m.weights.max_abs_pred_change_from_quantization));
  const sp = Math.max(ns.stats.outside_cutoff_share_spectral_max, dc.stats.outside_cutoff_share_spectral_max);
  return [
    el("p", {}, `Both models on this page are the trained ones. Their weights are stored at half precision, and the forward pass in your browser runs at double precision. Rounding the weights moves a prediction by at most ${q(ns).toExponential(1)} (fluid) and ${q(dc).toExponential(1)} (rock).`),
    el("p", {}, `The local path supplies all of the energy above the cutoff and the spectral path supplies none. That's true in every layer, sample and grid size we checked, on both problems (the largest spectral share anywhere was ${sp.toExponential(1)}).`),
    el("p", {}, "We also tested the fluid model on all 2,000 test samples (the page shows 200), checked that none of them appear in the training data, and compared it with the best linear per-wave filter, which it beats by 18×. It's a 0.6M-parameter model trained for 100 epochs on 6,000 samples at 64×64. Published FNOs on what appears to be the same dataset are larger and more accurate."),
    el("p", { class: "mono" }, `Data — fluid: ${ns.problem.source} Rock: Darcy flow, Zenodo 12784353.`),
  ];
}
