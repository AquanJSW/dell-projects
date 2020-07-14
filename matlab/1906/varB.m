%生成类间方差 varB(k)
function sigma_b_powered = varB(k)
    global mG;
    if (0 < P1(k)) && (P1(k) < 1)
        sigma_b_powered = (mG * P1(k) - graymean(k)).^2 / (P1(k) * (1 - P1(k)));
    else
        sigma_b_powered = 0;
    end
end
