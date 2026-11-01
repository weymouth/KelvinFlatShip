# No one cares about your research (yet!) — outline

30 min incl. Q&A → ~22 min talking. Audience: MSc/PhD students, no wave knowledge.
Running example: the JFM paper. Each section opens and closes with a meta slide; most set a "before" (the paper as written) against the "after" (the talk version).
Slide titles are headlines: a full sentence that states the point.

## Opening: the problem
1. **No one cares about your research (yet!)** — Title.
    *Graphic:* [Paper] graphical_abstract.jpg full-bleed, title set over the calm water to the right of the ship.
2. **You don't care about my latest paper (yet)** — The JFM abstract, untouched, full screen. "Hands up if you care." Nobody does, and that's the honest starting point for every talk. *(before)*
    *Graphic:* [Custom HTML] The abstract typeset as live text (not a screenshot), so slide 22 can reuse it with highlights.
3. **Your audience is smart, busy, and not you** — The curse of knowledge: after years on a problem you can't remember not knowing it. The jargon, the motivation and the "obvious" next step all live in your head, not theirs.
    *Graphic:* [Custom SVG] The tappers and listeners experiment (Newton, Stanford 1990): tappers predicted listeners would name 50% of the songs; they named 2.5% (3 of 120). Two bars.

## Motivation: make them care, then funnel down
4. **Open wide with a reason to care, then funnel down to your work** — Meta slide, no paper yet. The funnel: a hook the whole room already cares about → narrow to the problem → your contribution → widen back out to what it changes. Rules for the wide end: make it concrete (dates, costs, tallies, not "important" or "widely used"); pick the hook for this audience, not for you; lead with one hook; the conclusion can widen past it. Contrast: the paper's introduction starts at the narrow end, with Green's functions and citations.
    *Graphic:* [Custom SVG] An hourglass: hook → problem → contribution → what changes. It returns on every meta slide as a 'you are here' map with the current section lit.
5. **Making waves burns fuel** — Hook option: everyone. Shipping burns ~300 Mt of fuel a year, ~3% of global CO₂ (IMO, 2020); for a fast ship, wave-making can be around half the resistance. Predict the waves, and you can design them away. *(numbers to verify)*
    *Graphic:* [Commons] A ship throwing a big bow wave (fast ferry or naval vessel), with one big number: ~3% of global CO₂.
6. **Our physics models are built from points, and points blow up** — Hook option: physicists. Point mass (Newton, 1687), point charge (Coulomb, 1785), point vortex (Helmholtz, 1858), moving wave source (Kelvin, 1887): each idealisation is infinite at the point, and we learn to live with it. Kelvin's is different: at the water surface its infinity doesn't stay put but fills the whole wake (Ursell, 1960). Is the simplest wave theory wrong, or is the point the problem?
    *Graphic:* [Custom SVG + Paper] Four icons with dates (mass, charge, vortex, Kelvin source), each with a spike at the point; the last one swaps to pointsource_surface.png, its spike spreading down the wake.
7. **Kelvin developed stationary phase to explain ship waves** — Hook option: mathematicians. His 1887 method now runs through optics, quantum mechanics and seismology, yet his own integral is still the slow, unreliable case at the surface. Why is the problem that launched the method still the hard one? (Sets up slide 14.)
    *Graphic:* [Commons + Custom SVG] Kelvin's portrait (public domain) beside a stationary-phase sketch: a wiggly curve whose wiggles cancel everywhere except one flat spot.
8. **Design, control and AI need wave predictions in microseconds** — Hook option: engineers. A hull optimisation or an ML training set needs 10⁴–10⁶ evaluations; a CFD run takes hours on a cluster; an autopilot gets milliseconds. Each answer must be fast and must never blow up.
    *Graphic:* [Custom SVG] One log-time axis: CFD run (hours) · brute force (~2 s) · autopilot budget (ms) · our method (30 µs).
9. **Same research, four hooks: pick the one your audience already cares about** — Meta slide. The four options side by side, each with its audience. For this talk, the funnel continues from fuel.
    *Graphic:* [Reuse] Thumbnails of slides 5–8, each labelled with its audience; fuel lit.

## Theory: explain the mechanism, not the derivation
10. **Any sufficiently unexplained result is indistinguishable from magic** — Meta slide. Too much theory loses everyone; too little makes your result magic. The dial between "lecture" and "magic trick". A technical audience only trusts a result when they can see *why* it works. Rule of thumb: one mechanism per result, one picture per mechanism. Equations are fine when they clear up the mechanism, not when they prove how smart you are or load up information nobody needs yet.
    *Graphic:* [Custom SVG] A dial from 'lecture' to 'magic trick', needle in the middle; hourglass map in the corner.
11. **Build any ship from tiny disturbances, each making its own wake** — The theory in one picture: Kelvin's single-point wake is the building block; add many up and you get a ship. (This is the Green's function, named only after it's shown.)
    *Graphic:* [New Julia render + Custom] A single point-source wake (top view), then the same wake copied along a hull outline and summed into a ship's wake.
12. **A disturbance right at the surface makes infinitely short, infinitely steep waves** — The problem: the building block breaks exactly where a ship touches the water. Known since 1960, usually patched with fake damping.
    *Graphic:* [Paper, restyled] wavecut.png: the surface profile going from smooth to wild spikes as the source rises to the surface (blue curves only, legend replaced by a depth label).
13. **Everything cancels, but slowly** — The problem, both sides. Physics: two points side by side make long waves that add and short waves that cancel, so a real ship's width kills the noise, but cancelling ever-shorter waves needs ever more points. Numerics: each answer is an integral of fast wiggles that almost cancel; brute force must count every wiggle, and at the surface there are infinitely many.
    *Graphic:* [Custom JS] Two panels. Physics: two sources side by side, a long wave adding and a short wave cancelling. Numerics: the integrand plotted live, its wiggles stretching on forever as the depth slider goes to zero.
14. **So do the cancelling analytically** — The solution, both sides. Physics: integrate across the width exactly, with the elliptic loading of a finite wing, giving J₁(bk)/k: short waves die off and the wake is clean and finite (wave spectrum: point source growing without bound vs flat ship decaying; the wake comparison is saved for the demo). Numerics: only a few stationary points matter; compute those directly and send the rest through the complex plane, where the wiggles turn into smooth decay. ~30 µs instead of seconds.
    *Graphic:* [Paper, restyled + Custom SVG] Physics: spectrum.png, point source climbing vs flat ship decaying, with J₁(bk)/k beside it. Numerics: the complex plane, wiggles on the real axis vs a smooth path dipping below it.
15. **Every result got a "why", none got a proof** — Meta slide. Two contributions, two mechanisms, one equation that earned its place. Ask yourself: for each result in your talk, can the audience say in one sentence why it works?
    *Graphic:* [Custom] Two rows, result → why, in one line each; hourglass map.

## Results: show, don't tell
16. **Show the result, then say it in the title** — Meta slide. "Fast", "robust", "significantly better" are claims; a demo, a side-by-side or a number in a familiar unit is an experience. And each result's title is its headline: the one sentence to remember, because the audience reads the title first and may read nothing else.
    *Graphic:* [Custom] Hourglass map; the words 'fast', 'robust', 'better' crossed out beside a stopwatch.
17. **Live: put the ship anywhere, at any speed, and the wake keeps up** — Demo, in beats. Speed: same sliders, brute force stutters while ours redraws live (the told version was the abstract's "10⁴–10⁵ speedup"). Robustness: drag the ship onto the surface; the point-source wake turns to noise, the flat ship stays clean. Physics: sweep speed and width and watch each corner launch its own wave train, with a rooster tail behind the stern.
    *Graphic:* [Custom JS] The live demo (approach to decide: see below).
18. **Widening the ship cancels its angled waves, and the textbook theory misses it** — The key width finding: wave fields for a narrow, medium and wide ship on the same colour scale, the angled (diverging) waves fading as the ship widens; then wave drag falling with width, where classical thin-ship theory predicts no change. It's slide 13's cancellation, now visible in the result. Optional beat (cut if running long): drag also rises and falls with speed as bow and stern waves interfere.
    *Graphic:* [Paper + New Julia render] wave_field_b0.5 / b1 / b2 side by side on one colour scale; then drag vs width (a slice of wave_drag.png's data) with thin-ship theory as a flat line. Optional beat: drag vs length, the bow/stern humps and hollows.
19. **A result convinces when it's a comparison that pays off a mechanism** — Meta slide. No number in this section stood alone: brute force vs ours, point source vs flat ship, thin-ship vs flat-ship theory. And each one paid off a "why" from the theory section, so it felt earned, not magic. Last beat: every slide title so far was a headline; read them alone and you get the story. Ask yourself: what is each of your results compared against, and which mechanism does it pay off?
    *Graphic:* [Custom] Three 'X vs Y → mechanism' rows with thumbnails from 17–18; then the slide titles alone, listed down the screen.

## Conclusion: widen the funnel, cut the noise
20. **End wide, and cut everything that isn't the story** — Meta slide. The funnel's wide end: go back to the hook and say what changes. Conclusions attract noise: caveats, future-work lists, a recap of every result, thank-you slides, logos. Before: the paper's conclusion, one paragraph of findings, two of limits and extensions, and the payoff in its very last sentence. Tally: the paper has ~30 equations, 5 figures and 17 references; this talk used one equation.
    *Graphic:* [Custom SVG] Tally: paper (~30 equations, 5 figures, 17 references) vs talk (1 equation), as rows of icons; hourglass map.
21. **Wakes with width are finite, fast, and ready for applications** — The one conclusion slide, widening back out to the hooks. The infinity was the idealisation, not the physics: no fake damping needed. Kelvin's own integral now costs ~30 µs anywhere, so a million wave predictions take 30 seconds (vs ~3 weeks brute force, a century of CFD): enough for an optimiser, an autopilot or an ML training set. And it already says something about fuel: wave drag rises and falls with speed and drops as hulls widen.
    *Graphic:* [Paper] wave_field_b1 full-bleed, with three big numbers: finite everywhere · 30 µs · a million predictions in 30 s.
22. **Now you might care about my latest paper** — Meta slide. Slide 2's abstract, untouched. Now the room can read it, and some of them care. That's the "(yet!)". Last bullet: your turn, one sentence on why anyone should care about your work. Then Q&A.
    *Graphic:* [Reuse] Slide 2's abstract, with the phrases the talk explained now highlighted; 'your turn' as the last bullet.
