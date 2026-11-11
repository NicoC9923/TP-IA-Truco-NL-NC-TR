# A project-owned card deck replaces cards cut from a template image

Status: accepted (user decision, 2026-11-11).

## Context

The cards shown today are cut from a template image of unknown origin (`proyecto/generar_baraja.py` reads `Naipes_Españoles_Template.jfif` or `template_naipes.jpg`). Its licence is unknown, and the project will be shown in public. The new 3D table also needs card faces, and it should show the same cards as the web page.

## Decision

The 40-card Spanish deck is drawn by project code, so the project owns every card. One deck feeds both the web UI and the MuJoCo 3D scene, so a card looks the same in both. The template-cutting path and its image files are retired once the new deck is in use.

## Considered options

- **Keep the template cards:** no work, but the licence stays unknown for a public presentation.
- **Scan or download a commercial deck:** rejected; its artwork is copyrighted.

## Consequences

- No third-party artwork is shipped, so there is nothing to credit or clear.
- The drawn deck is simpler than traditional artwork. Cards stay recognisable by suit and number, which is what play needs.
- Any change to the deck's look is made once and applies to the web page and the 3D table.
