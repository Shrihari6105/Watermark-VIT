import numpy as np
import pytest

from watermark_vit import WaterMarkVIT, attacks, ecc, fragile, metrics, robust, tardos, templates


@pytest.fixture(scope="module")
def system(tmp_path_factory):
    w = WaterMarkVIT(tmp_path_factory.mktemp("data"))
    code = w.reg.create_exam(w.master, "TEST", [f"23BCE{1000 + i}" for i in range(100)])
    cover = w.prepare_cover(templates.answer_sheet(1))
    return w, code, cover


def test_ecc_roundtrip_and_correction():
    bits = ecc.encode(513, 37)
    soft = 2.0 * bits - 1
    soft[[0, 9, 20]] *= -1                       # three bit errors in three bytes
    dec = ecc.decode_soft(soft)
    assert (dec["roster_idx"], dec["exam_code"]) == (513, 37)


def test_decoder_rejects_noise(system):
    # Bare RS sometimes "decodes" noise; the full decoder adds a codeword-correlation gate
    w, _, _ = system
    rng = np.random.default_rng(0)
    rs_hits = sum(ecc.decode_soft(rng.normal(size=64)) is not None for _ in range(500))
    full_hits = sum(w.decode_id(rng.normal(size=64)) is not None for _ in range(500))
    assert rs_hits < 25
    assert full_hits == 0


def test_imperceptibility(system):
    w, code, cover = system
    wm = w.watermark_for(cover, code, 5)
    assert metrics.psnr(cover, wm) > 39
    assert metrics.ssim(cover, wm) > 0.97


@pytest.mark.parametrize("channel", ["jpeg50", "whatsapp", "rotate3", "printscan300", "phone30deg40cm", "crop50"])
def test_attribution_survives_channel(system, channel):
    w, code, cover = system
    wm = w.watermark_for(cover, code, 42)
    leak = {**attacks.DIGITAL, **attacks.PHYSICAL}[channel](wm, seed=1)
    r = w.investigate(leak, log=False)
    assert r["decoded"] and r["decoded"]["roster_idx"] == 42
    assert r["verdict"].startswith("Leak traced")


def test_unmarked_page_accuses_nobody(system):
    w, code, cover = system
    for leak in (cover, attacks.jpeg(cover, 70), attacks.crop(cover, 0.5)):
        r = w.investigate(leak, log=False)
        assert r["decoded"] is None
        assert not any(s["accused"] for s in (r.get("tardos") or {}).get("suspects", []))


def test_collusion_catches_a_colluder_only(system):
    w, code, cover = system
    a, b = w.watermark_for(cover, code, 10), w.watermark_for(cover, code, 77)
    r = w.investigate(attacks.jpeg(attacks.collude_average(a, b), 70), log=False)
    accused = {s["roster_idx"] for s in r["tardos"]["suspects"] if s["accused"]}
    assert accused and accused <= {10, 77}


def test_tardos_innocent_scores_are_centered():
    master = b"k" * 32
    X, p = tardos.codebook(master, "t", 500, 320, 2)
    y = X[0].astype(float) * 2 - 1
    sc, eff = tardos.scores(X, p, y)
    assert sc[0] > tardos.threshold(500, eff, 1e-3)
    assert abs(np.mean(sc[1:])) < 3 * np.sqrt(eff) / np.sqrt(499) + 5


def test_fragile_seal_verify_and_localise(system):
    w, code, cover = system
    page = templates.add_handwriting(w.watermark_for(cover, code, 3), seed=0, marks=[4, 6, 5])
    sealed, sid = w.seal(page)
    assert w.verify(sealed)["intact"]
    forged = attacks.tamper_marks(sealed, col=1, new_mark="9")
    res = w.verify(forged, script_id=sid)
    truth = metrics.block_truth(sealed, forged)
    p, r, f, _ = metrics.prf(res["tamper_map"], truth)
    assert f == 1.0 and res["tampered_blocks"] > 0


def test_fragile_key_binding(system):
    w, code, cover = system
    sealed = fragile.seal(cover, w.master, "E00-S0001")
    assert fragile.verify(sealed, w.master, "E00-S0001").sum() == 0
    assert fragile.verify(sealed, w.master, "E00-S0002").all()     # wrong script id fails everywhere


def test_leaktest_on_scanned_pages(tmp_path):
    import csv
    from watermark_vit import imaging, leaktest

    scans = tmp_path / "scans"
    # a "scanned" page: grey uneven paper, smaller than canonical, handwriting already on it
    page = templates.add_handwriting(templates.answer_sheet(seed=5), seed=3)
    page = (page.astype(float) * np.linspace(0.85, 0.95, page.shape[1])).astype(np.uint8)
    imaging.save(scans / "a.png", page[::2, ::2])
    imaging.save(scans / "b.pdf", page)
    leaktest.run([scans], tmp_path / "out", channels=["whatsapp", "phone30", "crop50"])
    rows = list(csv.DictReader(open(tmp_path / "out" / "results.csv")))
    assert len(rows) == 6 and all(r["outcome"] == "correct" for r in rows)
