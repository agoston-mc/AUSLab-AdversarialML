import AttackPack

for mname in AttackPack.methods.keys():

    AttackPack.main(
        method = mname,
        eps = [0.001, 0.0001],
        device = "cpu",
        weights = "vpnet-hermite2-rr_b4096_lr0.01_e50_p1.0-0.0_n8_h8_r0.0.pt",
        adv_method="d",
        # early_stop=1,
    )

