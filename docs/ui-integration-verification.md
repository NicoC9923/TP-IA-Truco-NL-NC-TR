# Spectator UI integration verification

Integrated `origin/feat/ia-entrenada` (including PR #49) and `origin/feat/ui-juego-espectador` into `feat/modo-espectador`; no PR was merged by this task.

## Conflict resolution

The base merge was clean. The UI merge conflicted in `proyecto/voz.html` (header controls and thinking lower third) and `proyecto/espectador.html` (match controls and visibility handlers). The resolution puts the speed and speech controls in Opciones, applies the new Más menu, and preserves shared `voz_robots.js`, asynchronous thinking/acting, speech toggles, speed handlers, hidden envido points, short matches, camera controls, and the rematch/poll-generation fixes.

## Fixes

- Human turn label gives `robot.ocupado` priority and says “Esperá”, including while a pending response exists; the active human styling also waits until the robot finishes.
- Opciones contains the labeled ×0,5 / ×1 / ×2 selector beside Voz de Ottoman, with 16 px text; all three buttons were checked for the correct POST body to `/truco/ritmo`.
- Spectator hand rows can wrap and cards use their intrinsic flex basis, reserving space for the lower thinking chip or moving it onto a complete row.

## Browser verification

Headless Microsoft Edge used `--mute-audio` and a context init script replacing `window.speechSynthesis.speak` with a no-op. Screenshots replay actual server snapshots to keep transient thinking states stable, including a spectator state after the first completed trick. Desktop viewport: 1440×900; mobile viewport: 390×844; screenshots include the whole page when vertical scrolling is needed.

Every requested screen had document width equal to viewport width (1440 or 390), with no browser JavaScript errors. The lower chip's bounding box stayed inside the table: desktop x=938.58, width=120.42, table right=1076; mobile x=29, width=120.42, table right=378. Inline JavaScript syntax, duplicate static IDs, and `git diff --check` also passed. The Impeccable detector ran in degraded regex mode and reported inherited layout-animation warnings; browser checks supplied the layout evidence.

Screenshots in `docs/screenshots/ui/` (both desk and mob):

- `voz-desk-1-inicio.png`, `voz-mob-1-inicio.png`
- `voz-desk-2-pensando.png`, `voz-mob-2-pensando.png`
- `voz-desk-4-opciones.png`, `voz-mob-4-opciones.png`
- `esp-desk-1-selector.png`, `esp-mob-1-selector.png`
- `esp-desk-1b-selector-manos.png`, `esp-mob-1b-selector-manos.png`
- `esp-desk-2-partida.png`, `esp-mob-2-partida.png`
- `esp-desk-3-mas.png`, `esp-mob-3-mas.png`
- `esp-desk-4-final.png`, `esp-mob-4-final.png`

The screenshot server and an extra server started while checking CLI help were stopped. The server parses `--puerto=<port>` and `--sin-robot` directly rather than providing argparse help.

## Full suite

Command from the worktree root: `PYTHONPATH=proyecto py -m unittest discover -s proyecto -p "test*.py"` (PYTHONPATH set through PowerShell).

Passed: 824 tests in 346.704 seconds; OK.
