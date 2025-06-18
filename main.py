import AttackPack

AttackPack.main(
    method = "FGSM",
    eps = [0.0, 0.001],
    device = "cpu",
    weights = "vpnet-hermite2-rr_b4096_lr0.01_e50_p1.0-0.0_n8_h8_r0.0.pt",
    adv_method="d",
    # early_stop=1,
)

entry = AttackPack.DATABASE[2]

AttackPack.show_entry(entry)

