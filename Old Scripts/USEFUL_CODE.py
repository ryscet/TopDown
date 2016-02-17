#CODE TO BE USED+

epochs['att_corr'].average().pick_channels(['Fp1']).plot(gfp = True)

epochs['mot_corr'].average().pick_channels(['Oz']).plot(axes = ax)