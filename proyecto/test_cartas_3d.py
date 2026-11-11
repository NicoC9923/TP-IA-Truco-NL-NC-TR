"""Tests for the project-owned deck and the Truco cards on the 3D table."""

from __future__ import annotations

import importlib.util
import random
import tempfile
import time
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import robot  # noqa: F401  (puts entorno/ on sys.path, as the server does)
from sim import cartas as C
from sim.visor import aviso_sin_mujoco

import generar_baraja as gb
import truco_reglas as tr
from partida_truco import PartidaTruco

DIR_CARTAS = Path(__file__).resolve().parent / "cartas"
HAY_MUJOCO = importlib.util.find_spec("mujoco") is not None


class TestBaraja(unittest.TestCase):
    def test_genera_40_caras_y_el_dorso_con_los_nombres_de_la_web(self):
        with tempfile.TemporaryDirectory() as tmp:
            generadas = gb.generar_baraja_40(tmp)
            esperadas = {f"{v}_{p}.png" for p in tr.PALOS for v in tr.VALORES} | {"dorso.png"}
            self.assertEqual(set(generadas), esperadas)
            self.assertEqual({p.name for p in Path(tmp).iterdir()}, esperadas)
            from PIL import Image
            for nombre in esperadas:
                with Image.open(Path(tmp) / nombre) as img:
                    self.assertEqual(img.size, gb.TAMANO_CARTA, nombre)
        # Same aspect ratio the web lays the cards out with (262 / 400).
        self.assertEqual(gb.TAMANO_CARTA[0] * 400, gb.TAMANO_CARTA[1] * 262)

    def test_las_cartas_versionadas_son_las_de_la_baraja_propia(self):
        nombres = {p.name for p in DIR_CARTAS.glob("*.png")}
        self.assertEqual(len(nombres), 41)
        self.assertFalse(list(DIR_CARTAS.glob("*.jpg")), "no template images left")
        from PIL import Image
        for p in DIR_CARTAS.glob("*.png"):
            with Image.open(p) as img:
                self.assertEqual(img.size, gb.TAMANO_CARTA, p.name)

    def test_dibujo_determinista_y_cartas_distintas(self):
        a = gb.dibujar_carta(7, "Oro").tobytes()
        self.assertEqual(a, gb.dibujar_carta(7, "Oro").tobytes())
        self.assertNotEqual(a, gb.dibujar_carta(7, "Copa").tobytes())
        self.assertNotEqual(gb.dibujar_carta(10, "Basto").tobytes(),
                            gb.dibujar_carta(12, "Basto").tobytes())

    def test_carta_inexistente(self):
        with self.assertRaises(ValueError):
            gb.dibujar_carta(8, "Oro")


class TestTexturas(unittest.TestCase):
    def test_cada_carta_tiene_su_textura_y_material(self):
        cartas = C.todas_las_cartas()
        self.assertEqual(len(cartas), 40)
        for carta in cartas:
            archivo = C.nombre_textura(carta)
            self.assertEqual(archivo, gb.nombre_archivo(carta.valor, carta.palo))
            self.assertTrue((DIR_CARTAS / archivo).is_file(), archivo)
        self.assertEqual(C.nombre_textura(C.Carta(1, "Espada")), "1_Espada.png")
        self.assertEqual(C.nombre_material(C.Carta(12, "Copa")), "mat_12_Copa")
        self.assertEqual(C.nombre_material(None), "mat_dorso")

    def test_carta_desde_el_formato_del_juego(self):
        self.assertEqual(C.Carta.de({"valor": 3, "palo": "basto", "nombre": "x"}), C.Carta(3, "Basto"))
        self.assertEqual(C.Carta.de((1, "Espada")), C.Carta(1, "Espada"))
        for mala in ({"valor": 8, "palo": "Oro"}, {"valor": 1, "palo": "Corazon"}):
            with self.assertRaises(ValueError):
                C.Carta.de(mala)


class TestEscenaXML(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.robots import G1
        cls.ruta = Path(G1.ruta_escena())
        cls.raiz = ET.parse(cls.ruta).getroot()

    def test_cartas_texturadas(self):
        texturas = {t.get("name"): t.get("file") for t in self.raiz.iter("texture")
                    if t.get("type") == "2d" and t.get("file")}
        self.assertEqual(len([n for n in texturas if n.startswith("tx_")]), 41)
        for nombre, archivo in texturas.items():
            self.assertTrue((self.ruta.parent / archivo).resolve().is_file(), archivo)
            self.assertFalse(Path(archivo).is_absolute(), "keeps the generated XML portable")
        materiales = {m.get("name") for m in self.raiz.iter("material")}
        for carta in C.todas_las_cartas():
            self.assertIn(C.nombre_material(carta), materiales)
        cuerpos = {b.get("name"): b for b in self.raiz.iter("body")}
        for asiento in C.ASIENTOS:
            for i in range(C.CARTAS_POR_ASIENTO):
                b = cuerpos[C.cuerpo(asiento, i)]
                self.assertEqual(b.get("mocap"), "true")
                geoms = {g.get("name"): g for g in b.iter("geom")}
                self.assertEqual(geoms[f"{C.cuerpo(asiento, i)}_frente"].get("mesh"), "malla_carta")
                self.assertEqual(geoms[f"{C.cuerpo(asiento, i)}_dorso"].get("material"), "mat_dorso")
        mallas = {m.get("name"): m for m in self.raiz.iter("mesh")}
        self.assertIn("texcoord", mallas["malla_carta"].attrib)

    def test_camaras_y_el_asiento_de_enfrente(self):
        camaras = {c.get("name") for c in self.raiz.iter("camera")}
        self.assertEqual(camaras, {"general", "detras_robot_1", "detras_robot_2", "mesa"})
        self.assertIn("silla_asiento_2", {b.get("name") for b in self.raiz.iter("body")})

    @unittest.skipUnless(HAY_MUJOCO, "MuJoCo no instalado")
    def test_la_escena_carga_en_mujoco(self):
        import mujoco
        m = mujoco.MjModel.from_xml_path(str(self.ruta))
        self.assertEqual(m.ncam, 4)
        self.assertEqual(m.nmocap, 6)
        for n in ("mano_izq", "mano_der"):
            self.assertGreaterEqual(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, n), 0)


MANO = [C.Carta(1, "Espada"), C.Carta(7, "Oro"), C.Carta(12, "Copa")]


def _correr(mesa: C.MesaCartas, segundos: float, dt: float = 0.02) -> list:
    fases = []
    for _ in range(round(segundos / dt)):
        mesa.avanzar(dt)
        j = mesa.jugada_activa(1)
        if j and (not fases or fases[-1] != j[0]):
            fases.append(j[0])
    return fases


class TestJugarCarta(unittest.TestCase):
    def setUp(self):
        self.mesa = C.MesaCartas()
        self.mesa.repartir({1: MANO, 2: [C.Carta(4, "Copa")] * 3})

    def por_carta(self, carta):
        return next(c for c in self.mesa.estado() if c.carta == carta)

    def test_repartir_pone_las_tres_en_la_mano_del_robot(self):
        en_mano = [c for c in self.mesa.estado() if c.lugar == "mano"]
        self.assertEqual([c.carta for c in en_mano], MANO)
        self.assertEqual([c.indice_abanico for c in en_mano], [0, 1, 2])
        # The human's cards are unknown to the scene until played.
        self.assertTrue(all(c.lugar == "oculta" for c in self.mesa.estado() if c.asiento == 2))

    def test_de_la_mano_a_la_mesa_con_la_carta_correcta_en_su_lugar(self):
        self.mesa.jugar(1, {"valor": 7, "palo": "Oro"})
        r = self.mesa.ritmo
        fases = _correr(self.mesa, r.alcanzar + r.llevar + r.volver + 0.1)
        self.assertEqual(fases, ["alcanzar", "llevar", "volver"])
        c = self.por_carta(C.Carta(7, "Oro"))
        self.assertEqual((c.lugar, c.baza, c.asiento), ("mesa", 0, 1))
        self.assertFalse(self.mesa.ocupado())
        restantes = [c for c in self.mesa.estado() if c.lugar == "mano"]
        self.assertEqual([c.carta for c in restantes], [MANO[0], MANO[2]])
        self.assertEqual([c.indice_abanico for c in restantes], [0, 1])

    def test_la_carta_viaja_durante_llevar(self):
        self.mesa.jugar(1, MANO[0])
        r = self.mesa.ritmo
        _correr(self.mesa, r.alcanzar + r.llevar / 2)
        c = self.por_carta(MANO[0])
        self.assertEqual((c.lugar, c.fase), ("viaje", "llevar"))
        self.assertAlmostEqual(c.progreso, 0.5, delta=0.05)

    def test_bazas_siguientes_usan_el_siguiente_lugar(self):
        for carta in (MANO[2], MANO[0], MANO[1]):
            self.mesa.jugar(1, carta)
        _correr(self.mesa, 3 * 1.8)
        self.assertEqual([self.por_carta(c).baza for c in (MANO[2], MANO[0], MANO[1])], [0, 1, 2])
        self.assertTrue(all(c.lugar == "mesa" for c in self.mesa.estado() if c.asiento == 1))
        posiciones = {C.slot_mesa(1, b)[0] for b in range(3)}
        self.assertEqual(len(posiciones), 3)

    def test_carta_del_humano_aparece_de_su_lado(self):
        self.mesa.jugar(2, {"valor": 3, "palo": "Basto"})
        _correr(self.mesa, self.mesa.ritmo.deslizar + 0.05)
        c = self.por_carta(C.Carta(3, "Basto"))
        self.assertEqual((c.lugar, c.asiento, c.baza), ("mesa", 2, 0))
        (x2, _, _), _ = C.slot_mesa(2, 0)
        (x1, _, _), _ = C.slot_mesa(1, 0)
        self.assertGreater(x2, C.MESA_X)
        self.assertLess(x1, C.MESA_X)

    def test_jugadas_en_cola_no_se_pierden(self):
        self.mesa.jugar(1, MANO[0])
        self.mesa.jugar(1, MANO[1])
        self.assertTrue(self.mesa.ocupado())
        _correr(self.mesa, 2 * 1.8)
        self.assertEqual(self.por_carta(MANO[1]).baza, 1)
        self.assertEqual(self.por_carta(MANO[1]).lugar, "mesa")

    def test_repartir_limpia_la_mesa(self):
        self.mesa.jugar(1, MANO[0])
        self.mesa.jugar(2, C.Carta(3, "Basto"))
        _correr(self.mesa, 0.5)
        otra = [C.Carta(2, "Oro"), C.Carta(5, "Copa"), C.Carta(6, "Basto")]
        self.mesa.repartir({1: otra})
        self.assertFalse(self.mesa.ocupado())
        self.assertEqual([c.carta for c in self.mesa.estado() if c.lugar == "mano"], otra)
        self.assertFalse([c for c in self.mesa.estado() if c.lugar in ("mesa", "viaje")])

    def test_carta_que_no_estaba_en_la_mano_igual_se_muestra(self):
        self.mesa.jugar(1, C.Carta(5, "Espada"))
        _correr(self.mesa, 1.0)
        c = self.por_carta(C.Carta(5, "Espada"))
        self.assertEqual((c.lugar, c.baza), ("mesa", 0))

    def test_duracion_configurable(self):
        mesa = C.MesaCartas(C.Ritmo(alcanzar=0.1, llevar=0.2, volver=0.1))
        mesa.repartir({1: MANO})
        mesa.jugar(1, MANO[0])
        _correr(mesa, 0.45)
        self.assertFalse(mesa.ocupado())


class TestCamaras(unittest.TestCase):
    def test_contra_un_humano_no_hay_camara_detras_del_robot(self):
        mesa = C.MesaCartas()
        self.assertEqual(mesa.modo, C.MODO_HUMANO)
        self.assertEqual(C.camaras_permitidas(C.MODO_HUMANO), (C.CAMARA_GENERAL,))
        for nombre in ("detras_robot_1", "detras_robot_2"):
            self.assertEqual(mesa.elegir_camara(nombre), "general")
        self.assertEqual(C.camara_siguiente("general", C.MODO_HUMANO), "general")
        # And whatever the viewer shows, the robot's hand shows only backs.
        self.assertTrue(mesa.ocultar_caras_en_mano(1))

    def test_robot_contra_robot_habilita_todas(self):
        mesa = C.MesaCartas()
        mesa.poner_modo(C.MODO_ROBOTS)
        self.assertEqual(mesa.elegir_camara("detras_robot_1"), "detras_robot_1")
        self.assertEqual(C.camara_siguiente("detras_robot_1", C.MODO_ROBOTS), "detras_robot_2")
        self.assertEqual(C.camara_siguiente("detras_robot_2", C.MODO_ROBOTS), "mesa")
        self.assertEqual(C.camara_siguiente("mesa", C.MODO_ROBOTS), "general")
        self.assertFalse(mesa.ocultar_caras_en_mano(1))
        # Back to a human: the behind camera turns itself off.
        mesa.poner_modo(C.MODO_HUMANO)
        self.assertEqual(mesa.camara, "general")

    def test_nombres_y_modos_invalidos(self):
        mesa = C.MesaCartas()
        with self.assertRaises(ValueError):
            mesa.elegir_camara("cenital")
        with self.assertRaises(ValueError):
            mesa.poner_modo("todos")

    def test_ordenes_del_socket(self):
        from sim.local import ServidorLocal
        from sim.mundo import Mundo
        from sim.robots import G1
        mundo = Mundo()
        mesa = C.MesaCartas()
        srv = ServidorLocal(mundo, G1, mundo.limites, verboso=False, puerto=0, cartas=mesa)
        try:
            r = srv.despachar({"orden": "camara", "nombre": "detras_robot_1"})
            self.assertEqual(r, {"ok": True, "camara": "general"})
            self.assertTrue(srv.despachar({"orden": "cartas_repartir",
                                           "manos": {"1": [c.como_dict() for c in MANO]}})["ok"])
            self.assertTrue(srv.despachar({"orden": "cartas_jugar", "asiento": 2,
                                           "carta": {"valor": 3, "palo": "Basto"}})["ok"])
            self.assertTrue(mesa.ocupado())
            self.assertEqual(srv.despachar({"orden": "modo_mesa", "modo": "robots"})["camara"], "general")
            r = srv.despachar({"orden": "camara", "nombre": "detras_robot_1"})
            self.assertEqual(r["camara"], "detras_robot_1")
        finally:
            srv.server_close()
        sin = ServidorLocal(mundo, G1, mundo.limites, verboso=False, puerto=0)
        try:
            self.assertFalse(sin.despachar({"orden": "camara", "nombre": "general"})["ok"])
        finally:
            sin.server_close()


@unittest.skipUnless(HAY_MUJOCO, "MuJoCo no instalado")
class TestEscenaMuJoCo(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from sim.capturas import Fotografo
        cls.f = Fotografo(320, 240)

    def setUp(self):
        self.f.mesa.poner_modo(C.MODO_HUMANO)
        self.f.mesa.repartir({1: MANO})
        self.f.correr(0.5)

    def _cuerpo(self, carta):
        return next(c for c in self.f.mesa.estado() if c.carta == carta).cuerpo

    def test_la_carta_jugada_queda_en_su_lugar_con_su_cara(self):
        import numpy as np
        f = self.f
        f.mesa.jugar(1, MANO[1])
        r = f.mesa.ritmo
        f.correr(r.alcanzar + r.llevar + r.volver + 0.2)
        e = f.sim.cartas
        n = self._cuerpo(MANO[1])
        pos, _ = C.slot_mesa(1, 0)
        np.testing.assert_allclose(f.sim.data.mocap_pos[e.mocap[n]], pos, atol=1e-6)
        self.assertEqual(f.sim.model.geom_matid[e.geom_frente[n]], e.material[MANO[1]])

    def test_en_modo_humano_la_mano_del_robot_muestra_dorsos(self):
        e = self.f.sim.cartas
        for carta in (MANO[0], MANO[2]):
            self.assertEqual(self.f.sim.model.geom_matid[e.geom_frente[self._cuerpo(carta)]],
                             e.material_dorso)
        self.f.mesa.poner_modo(C.MODO_ROBOTS)
        self.f.correr(0.05)
        self.assertEqual(self.f.sim.model.geom_matid[e.geom_frente[self._cuerpo(MANO[0])]],
                         e.material[MANO[0]])

    def test_las_cartas_en_mano_siguen_a_la_mano_izquierda(self):
        import numpy as np
        e, d = self.f.sim.cartas, self.f.sim.data
        palma = d.site_xpos[e.sitio_izq]
        for carta in MANO:
            centro = d.mocap_pos[e.mocap[self._cuerpo(carta)]]
            self.assertLess(np.linalg.norm(centro - palma), 0.12)
            self.assertGreater(centro[2], C.MESA_Z + 0.1)       # above the table

    def test_el_visor_nunca_queda_en_una_camara_prohibida(self):
        import mujoco
        e = self.f.sim.cartas
        cam = mujoco.MjvCamera()
        e.sincronizar_visor(cam)
        self.assertEqual(cam.fixedcamid, e.camaras["general"])
        cam.fixedcamid = e.camaras["detras_robot_1"]          # e.g. MuJoCo's own ] key
        e.sincronizar_visor(cam)
        self.assertEqual(cam.fixedcamid, e.camaras["general"])
        e.tecla(C.TECLA_CAMARA)
        e.sincronizar_visor(cam)
        self.assertEqual(cam.fixedcamid, e.camaras["general"])


class TestAvisoSinMuJoCo(unittest.TestCase):
    def test_aviso_en_castellano_con_el_comando(self):
        aviso = aviso_sin_mujoco(buscar=lambda nombre: None)
        self.assertIn("falta MuJoCo", aviso)
        self.assertIn("instalar.bat", aviso)
        self.assertIn("pip install", aviso)
        self.assertIn("mujoco==3.12.0", aviso)
        self.assertIsNone(aviso_sin_mujoco(buscar=lambda nombre: object()))

    def test_el_chequeo_del_entorno_dice_como_instalarlo(self):
        from sim import verificar
        with patch.object(verificar.importlib, "import_module",
                          side_effect=ImportError("No module named 'mujoco'")):
            ok, porque = verificar.diagnosticar_mujoco()
        self.assertFalse(ok)
        self.assertIn("instalar.bat", porque)
        self.assertIn("mujoco==3.12.0", porque)


class TestPartidaMueveLasCartas3D(unittest.TestCase):
    def test_reparto_y_jugadas_llegan_a_la_mesa_en_orden(self):
        with tempfile.TemporaryDirectory() as tmp:
            robot_falso = MagicMock()
            random.seed(1)
            partida = PartidaTruco(robot=robot_falso, historial_path=str(Path(tmp) / "h.json"))
            self.assertTrue(partida.es_mano_humano and partida.turno_carta == "humano")
            carta = partida.cartas_humano_disponibles[0]
            partida.jugar_carta_humano({"valor": carta["valor"], "palo": carta["palo"]})
            esperadas = 2 + (1 if any(m["quien"] == "robot" for m in partida.mesa) else 0)
            limite = time.monotonic() + 5
            while time.monotonic() < limite:
                llamadas = [c for c in robot_falso.mock_calls
                            if c[0] in ("repartir_cartas", "tirar_carta")]
                if len(llamadas) >= esperadas:
                    break
                time.sleep(0.02)
            self.assertEqual(llamadas[0][0], "repartir_cartas")
            self.assertEqual(llamadas[0][1][0], partida.cartas_robot)
            self.assertEqual(llamadas[1][0], "tirar_carta")
            self.assertEqual(llamadas[1][1], (carta, 2))
            jugada_robot = [m["carta"] for m in partida.mesa if m["quien"] == "robot"]
            if jugada_robot:
                self.assertEqual(llamadas[2][1], (jugada_robot[0], 1))

    def test_sin_simulador_la_partida_no_cambia(self):
        with tempfile.TemporaryDirectory() as tmp:
            partida = PartidaTruco(robot=None, historial_path=str(Path(tmp) / "h.json"))
            carta = partida.cartas_humano_disponibles[0]
            resultado = partida.jugar_carta_humano(dict(carta))
            self.assertNotIn("error", resultado)
            self.assertTrue(partida._cola_mesa.empty())


if __name__ == "__main__":
    unittest.main()


class TestServidorSeReconectaAlSimulador(unittest.TestCase):
    def test_reiniciar_el_simulador_reconstruye_la_mesa_3d(self):
        import servidor_truco

        class SimCaido(Exception):
            pass
        SimCaido.__name__ = "ErrorTransporte"

        with tempfile.TemporaryDirectory() as tmp:
            viejo = MagicMock()
            viejo.tirar_carta.side_effect = SimCaido("socket cerrado")
            partida = PartidaTruco(robot=viejo, historial_path=str(Path(tmp) / "h.json"))
            servidor = servidor_truco.ServidorTruco(("127.0.0.1", 0), partida, False)
            try:
                self.assertIs(servidor.asegurar_robot(), viejo)
                partida._mesa_3d("tirar_carta", {"valor": 1, "palo": "Espada"}, 2)
                limite = time.monotonic() + 5
                while servidor.robot is not None and time.monotonic() < limite:
                    time.sleep(0.02)
                self.assertIsNone(servidor.robot)
                self.assertIsNone(partida.robot)

                nuevo = MagicMock()
                with patch.object(servidor_truco, "Robot", return_value=nuevo):
                    self.assertIs(servidor.asegurar_robot(), nuevo)
                self.assertIs(partida.robot, nuevo)
                limite = time.monotonic() + 5
                while not nuevo.repartir_cartas.called and time.monotonic() < limite:
                    time.sleep(0.02)
                nuevo.modo_mesa.assert_called_with("humano")
                nuevo.repartir_cartas.assert_called_once()
            finally:
                servidor.server_close()
