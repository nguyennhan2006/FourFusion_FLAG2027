print("[cell] raw_evalphase: member-by-member check against the dataset the system read", flush=True)
# expected/eval_set_members.sha256 = sha256 of every member of the eval_set.zip uploaded as ngnnguynchn/flag2027-evalphase-p
# (D:/ff_data/evalphase/eval_set.zip, sha256 a70762a3..., 4 669 members; measured 18/11/2026).
EXP = {}
for line in (STAGE_DIR / "expected/eval_set_members.sha256").read_text(encoding="utf-8").splitlines():
    if line.strip():
        h, name = line.split("  ", 1)
        EXP[name] = h
with zipfile.ZipFile(EVAL_SET) as z:
    GOT = {i.filename: hashlib.sha256(z.read(i)).hexdigest() for i in z.infolist() if not i.filename.endswith("/")}
missing = sorted(set(EXP) - set(GOT))
extra = sorted(set(GOT) - set(EXP))
differ = sorted(n for n in set(EXP) & set(GOT) if EXP[n] != GOT[n])
REP.update(members=len(GOT), members_expected=len(EXP), missing=missing[:20], extra=extra[:20], differ=differ[:20],
           members_ok=not (missing or extra or differ))
print(f"members {len(GOT)} (expected {len(EXP)}): missing {len(missing)}, extra {len(extra)}, different {len(differ)}", flush=True)
(OUT / "raw_evalphase_check.json").write_text(json.dumps(REP, indent=1))
assert REP["members_ok"], f"eval_set.zip content differs from the dataset the system read: {missing[:3]} {extra[:3]} {differ[:3]}"
