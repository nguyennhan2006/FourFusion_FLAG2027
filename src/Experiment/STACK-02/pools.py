"""STACK-02 candidate families (registered 07/10 before any STACK-02 score). Each family lists its candidate columns; the
representatives are chosen INSIDE every outer fold (run.py). Excluded by name, before any score: control arms that re-create
a base stream or an untrained initialisation, the length-biased LOCAL-ID max streams (LOCALID-01 registration), and the
partially covered streams (acl, lb, iba2, ibal, s010B). `vox2` marks an identified VoxCeleb2-DATA dependency (trained or
fitted on VoxCeleb2 rows); a VoxCeleb/VoxBlink-pretrained checkpoint used as a frozen encoder is not marked.
"""
LAYERS = ([f"rel{g}S_{e}_L{l:02d}" for e, n in (("wlmbp", 13), ("wlmlg", 25), ("hubl", 25)) for l in range(n) for g in ("", "g")]
          + [f"rel{g}S_{e}" for e in ("wsr221", "wssim34") for g in ("", "g")])
POOLS = {
    "PE-AV":    dict(table="peav01", cols=["pe", "pe4", "peraw", "relS_pe", "relgS_pe"], vox2=False),       # - relS_base, relgS_base (= rel)
    "LOCAL-ID": dict(table="localid01", cols=["li_pm", "li_cm3", "li_mm3", "li_learn"], vox2=False),       # - li_glob (~ XALL_ib), li_cm, li_mm (length bias)
    "CFE":      dict(table="cfe01", cols=["A1_nce", "A1_cfe", "A2_nce", "A2_cfe"], vox2=False),
    "LAYER":    dict(table="layerid01", cols=LAYERS, vox2=False),                                           # rel / relg with WavLM base+, WavLM large, HuBERT large layers, WeSpeaker
    "LOSSG":    dict(table="lossg01", cols=["d_SCG100", "d_SCG50", "d_SCG50_SPX", "d_SPX"], vox2=False),   # - d_CTRL (= deep0), d_CTRL_D (dimension control)
    "RESADAPT": dict(table="resadapt01", cols=["ra", "ra_r64", "ra_wd1"], vox2=False),                    # - ra0 (= XALL_ib)
    "VOX2-BR":  dict(table="vox2a", cols=["d_XV1K", "d_XV5K", "d_XV5KL"], vox2=True),                       # - d_XALL (= XALL_deep0)
    "VOX2-AD":  dict(table="vox2b", cols=[f"ad_s{s}_a{a}" for s in (250, 500, 1000, 2000, 4000) for a in (0.25, 0.5, 1.0)], vox2=True),  # - ad0
    "CKPT":     dict(table="ckpt01", cols=["ck_cmpc", "ck_cmpc_raw", "ck_wen", "ck_wen_raw"], vox2=False),
    "ATTR":     dict(table="attrconf01", cols=["ovl_age"], vox2=False),
    "REL-ALL":  dict(table="relonly_multi", cols=["relm_fia_e", "relm_fia_eb", "relm_fi_e", "relc_f_e", "relc_fia_e", "relc_all_eb"], vox2=False),  # - relc_fia_eb (= rel)
}
TABLES = sorted({f["table"] for f in POOLS.values()} - {"attrconf01"})        # attrconf01 is loaded by stream_gate.load
K_REP = 2              # representatives per family (1 if the pool has one column)
R_MAX = 0.90           # a second representative must have |Pearson r| < R_MAX with the first on the training pool
C_STACK = 0.05         # logistic C of every extended stack; BASE-6 keeps CA-FUSE's registered C = 1
