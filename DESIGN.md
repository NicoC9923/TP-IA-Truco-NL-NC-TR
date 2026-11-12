# Design: Torneo Televisado

The Truco UI (`proyecto/voz.html`) is styled as a live TV broadcast of a Truco match. The audience is a classroom watching a projector, and it reads Ottoman's hidden reasoning the way poker broadcasts show the players' hidden cards. These notes describe what is built. The source is the `:root` tokens in `voz.html`.

## World

- **Scene:** a dimmed classroom with a projector, viewed from several meters away. The UI is dark only, and that is deliberate.
- **Material:** flat broadcast plates on deep studio navy. The lower-third, the canto and the final plate have slanted cuts made with `clip-path` and `--cut: 14px`; the scorebug is square-cornered. Shadows are minimal (`--shadow-1` 0 1px 2px, `--shadow-2` 0 2px 8px) and plates carry no drop-shadow filters. There is no glow, no glass and no gradient text. The only gradient is the radial "studio light" on the table shot.
- **Teams carry color:** Ottoman is scorebug yellow `--ott #ffcc29` and Vos is celeste `--vos #5cc6ff`. Every reference to a player uses that player's color: score, turn tag, baza winner, card slots, the lower-third tab.
- **Red is reserved** (`--live #f03c3c`) for EN VIVO (simulator connected), MENTIRA (a bluff), blocked commands and Al mazo.

## Tokens

| Role | Token | Value |
|---|---|---|
| Page | `--ink-0` | #040b1c |
| Plate | `--ink-1` / `--ink-2` / `--ink-3` / `--ink-4` | #08142e / #0e1f47 / #163063 / #1f3f7e |
| Lines | `--line` / `--line-soft` | #22407c / #172f5e |
| Text | `--txt` / `--txt-2` / `--txt-3` | #f2f5fb / #b3c1df / #8598c2 |
| Ottoman | `--ott`, `--ott-ink` | #ffcc29 on #1c1500 |
| Vos | `--vos`, `--vos-ink` | #5cc6ff on #00192b |
| Signals | `--live`, `--ok`, `--warn` | #f03c3c, #3fcf86, #ff9b3d |
| Red under small white text | `--live-deep` | #c42b2b (5.6:1 with white; e.g. Detenerse on hover) |

## Type

- **Display:** Sofia Sans Extra Condensed, 800–900, often italic and uppercase. Used for numerals, canto names, plate titles, buttons and tabs. The fonts are self-hosted in `proyecto/fuentes/` and served at `/fuentes/*.woff2`, so they work offline.
- **UI:** Sofia Sans Semi Condensed, 400–700.
- Tabular figures are used everywhere.
- Scale contrast does the work. The score is 50px and the live canto is up to 64px. Data labels are 11px tracked uppercase (11px is the floor for functional text).
- Italic and weight 900 are reserved for the score numerals, the canto and the end-of-match title. Everything else uses 600–800 upright: tabs, buttons, player names, plate titles.

## Components

- **Scorebug:** a Vos plate, an Ottoman plate, then an info plate, as plain rectangles separated by 2px gaps with 6px outer corners (no slanted cuts). The info plate holds the goal toggle (A 15 / A 30), progress tracks with a malas/buenas tick at 30, the stake name, and a 4-step stake ramp (1–4 points, fixed sizes).
- **Table shot:** Ottoman's row (face-down backs, flipped to faces at the end of a hand), then three baza frames (Ottoman's card on top, yours below, the winner tag, the active baza outlined in yellow), then the lower-third, then your hand, then the call row.
- **Lower-third (zócalo):** a yellow name tab and a white bar with Ottoman's line in quotes. Under the line sits its "proof": the decision badge, the metric and the threshold from `ultimo_razonamiento`. While Ottoman thinks it switches to navy with a scan bar.
- **Canto:** replaces the lower-third. A full-width yellow slab wipes in with the canto at display size, the stakes, and the response buttons (green for quiero, dark red for no quiero, black for raises).
- **Reveal (hole-card cam):** Ottoman's three cards flip face up. Cards it kept are outlined in yellow ("Se la guardó"). The verdict shows who won the hand, Ottoman's real tantos, the envido result, and each MENTIRA with its outcome (or "Jugó derecho").
- **Layout:** the table shot takes the page. A slim 220px strip on the left holds the inputs: the camera with its two buttons and status line, push-to-talk with a short phrase hint, and the text field. At 1100px and below the strip drops under the table.
- **Analysis drawer:** hidden by default. "Ver análisis" in the top bar opens it on the right, below the header, and the table shifts left so nothing is covered. Cerrar or Esc closes it. In order it holds:
  - Por qué jugó así (the decision trace)
  - Mentiras (bluff-quota meters, this match's bluff table, the read on your play)
  - Cerebro (agent nameplates; only the active one shows its description)
  - Relato (the play-by-play log)
- The lower-third's proof line keeps Ottoman's latest reasoning visible while the drawer is closed.
- **Final:** an overlay on the table shot only, never the whole page. It shows the winner's plate, the score and the bluff stats.
- **Control G1:** a large push-to-talk button, a language toggle, a written-order field, a 7-button console grid (Detenerse is red), and an order log with Hecho / Bloqueada / No entendí tags.

## Motion

- Nothing moves while the page is idle: no blinking live dot, no pulsing Repartir.
- One move per event:
  - The canto wipe (clip-path, 0.6s).
  - A played card settles in.
  - A score number ticks up.
  - The reveal flips cards in a stagger.
- Easing is `--ease-out` (expo).
- Everything is disabled under `prefers-reduced-motion`.

## Rules

- The current decision (play a card, answer a canto) is always in the table shot.
- At 1280×720 the Mesa fits without scrolling the page. The side columns scroll on their own. Card sizes come from `vh` clamps.
- Icons are an inline SVG sprite (24 grid, 1.8 stroke). Use no emoji.
- Copy is Rioplatense Spanish with voseo, using Truco terms as the server names them.
- Browser surfaces are themed: selection, caret, scrollbars, and a yellow focus ring.
