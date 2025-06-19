import AttackPack

eps_dat = [0.1, 0.01, 0.001, 0.0001]
eps_mix = [0.1, 0.01]

for mname in AttackPack.methods.keys():

    AttackPack.main(
        method = mname,
        eps = eps_dat,
        device = "cpu",
        weights = "vpnet-hermite2-rr_b4096_lr0.01_e50_p1.0-0.0_n8_h8_r0.0.pt",
        adv_method="d",
        # early_stop=1,
    )
    AttackPack.main(
        method=mname,
        eps=eps_mix,
        device="cpu",
        weights="vpnet-hermite2-rr_b4096_lr0.01_e50_p1.0-0.0_n8_h8_r0.0.pt",
        adv_method="r",
        # early_stop=1,
    )
    AttackPack.main(
        method=mname,
        eps=eps_mix,
        device="cpu",
        weights="vpnet-hermite2-rr_b4096_lr0.01_e50_p1.0-0.0_n8_h8_r0.0.pt",
        adv_method="dr",
        # early_stop=1,
    )

