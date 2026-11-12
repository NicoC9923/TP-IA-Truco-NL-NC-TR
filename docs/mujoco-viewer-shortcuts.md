# MuJoCo 3.12.0 viewer keyboard audit

Checked against the installed Python package (`mujoco.__version__ == "3.12.0"`),
including its exported `mjVISSTRING` and `mjRNDSTRING` tables, and the official
3.12.0 sources below. This audit covers the simulate viewer used by
`mujoco.viewer.launch_passive`, not other MuJoCo applications.

## Event ordering and camera key

The Python adapter calls the native adapter before the Python callback
([python/mujoco/simulate.cc, `OnKey`](https://github.com/google-deepmind/mujoco/blob/3.12.0/python/mujoco/simulate.cc#L61-L68)).
The native adapter dispatches the UI event before returning
([platform_ui_adapter.cc](https://github.com/google-deepmind/mujoco/blob/3.12.0/simulate/platform_ui_adapter.cc#L57-L80)).
Returning from our callback cannot consume or undo those shortcuts.

**F7 cycles `opt.label`**, modulo `mjNLABEL`: None, Body, Joint, Geom, and
the remaining label modes. It does not toggle the Joint visualization flag
(that is J). **F8 (GLFW 297) has no built-in binding**, including in UI editing,
so it now cycles our allowed cameras without changing visualization options.
F9-F12 are also unassigned. All plain A-Z keys have visualization/rendering
bindings, so a letter would introduce another collision.

## Global and UI shortcuts

Sources: [simulate.cc event handler](https://github.com/google-deepmind/mujoco/blob/3.12.0/simulate/simulate.cc#L1884-L2030),
[simulate.h UI definitions](https://github.com/google-deepmind/mujoco/blob/3.12.0/simulate/simulate.h#L313-L360),
and [simulate.cc UI construction](https://github.com/google-deepmind/mujoco/blob/3.12.0/simulate/simulate.cc#L111-L130).
Some simulation operations are disabled in passive mode; their keys are still
reserved here rather than reused by the application.

| Key | Built-in action |
| --- | --- |
| F1 / F2 / F3 / F4 | Help / info / profiler / sensors |
| F5 | Fullscreen (disabled on macOS) |
| F6 / F7 | Cycle coordinate frames / labels |
| Esc | Free camera |
| Tab / Shift+Tab | Toggle left / right UI panel |
| Space | Run/pause (non-passive only) |
| Left / Right | History backward / step forward (non-passive only) |
| Page Up | Select parent of selected body |
| `[` / `]` | Previous / next fixed camera |
| `-` / `=` | Slow down / speed up (unshifted, non-passive only; help describes `+`) |
| Backspace | Reset simulation |
| 0-5 / Shift+0-5 | Toggle geom / site groups |
| Ctrl+A / Ctrl+C / Ctrl+L | Align / copy state / reload |
| Ctrl+M / Ctrl+D / Ctrl+P / Ctrl+Q | Print model / print data / screenshot / quit |
| Alt+F/O/S/W/P/R/V/G/L | Open File/Option/Simulation/Watch/Physics/Rendering/Visualization/Group/Logging section, collapsing the others |
| Alt+J/C/E | Open Joint/Control/Equality section, collapsing the others |

There are no global Up, Down, Page Down, Home, End, Insert, Delete, Enter,
6-9 or F8-F12 bindings in those tables or the event switch. In a focused UI
edit field, Esc cancels, Enter commits, Left/Right/Home/End move the cursor
(Ctrl+Left/Right move to the start/end),
Backspace/Delete erase, and accepted printable/numpad keys enter text.
See [ui_main.c editing and shortcut dispatch](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/ui/ui_main.c#L2734-L2870).
The key-name display table includes F8-F12 but does not bind them.

## Visualization and rendering flags

The viewer constructs checkbox shortcuts from the installed/exported tables:
[engine_vis_init.c](https://github.com/google-deepmind/mujoco/blob/3.12.0/src/engine/engine_vis_init.c#L69-L119),
[MakeRenderingSection](https://github.com/google-deepmind/mujoco/blob/3.12.0/simulate/simulate.cc#L955-L1024).
These are plain, unmodified keys (the UI checks modifiers exactly).

| Keys | Visualization flags, in the same order |
| --- | --- |
| H, X, J, Q, U | Convex hull, texture, joint, camera, actuator |
| Z, V, Y, E, I | Light, tendon, range finder, equality, inertia |
| B, O, C, N, F, P | Perturb force, perturb object, contact point, island, contact force, contact split |
| T, A, M, D | Transparent, auto connect, center of mass, static body |
| comma, apostrophe, semicolon, backtick, backslash | Activation, scale inertia, skin, body tree, mesh tree |

| Keys | Rendering flags, in the same order |
| --- | --- |
| S, W, R, L, K, G | Shadow, wireframe, reflection, additive, skybox, fog |
| slash, comma | Haze, segment |

Comma is listed in both tables; UI shortcut search processes the first active
match. Flags with an empty shortcut have no keyboard binding. Camera requests
and `sincronizar_visor` only access the camera, leaving the user's label/frame,
option flags, groups, and rendering flags untouched. Human mode still repairs
forbidden fixed cameras after native bracket navigation; Esc remains free camera.
