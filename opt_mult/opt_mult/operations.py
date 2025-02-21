def xmymwh_xyxy(xm, ym, w, h):
    xtl = xm - w/2
    ytl = ym - h/2
    ybr = ym + h/2
    xbr = xm + w/2

    return xtl, ytl, xbr, ybr