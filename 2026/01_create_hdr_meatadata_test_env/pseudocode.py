def browser_hdr_to_hdr_conversion():
    hdr10_mdcv_luminance = get_hdr10_mdcv_luminance()  # 無ければ None を返す想定
    hdr10_clli_luminance = get_hdr10_clli_luminance()  # 無ければ None を返す想定
    sdr_content_brightness = get_sdr_content_brightness()  # Windows の UI 的に 80～480 nits の想定
    monitor_peak_luminance = get_monitor_peak_luminance()  # EDID / MHC Profile の仕様上 None は返さない想定
    hdr_content = get_hdr_content()

    if hdr10_clli_luminance is not None:
        hdr_content_luminance = hdr10_clli_luminance
    elif hdr10_mdcv_luminance is not None:
        hdr_content_luminance = hdr10_mdcv_luminance
    else:
        hdr_content_luminance = 10000

    modified_hdr_content = apply_tone_mapping(
        hdr_content=hdr_content,
        src_luminance=hdr_content_luminance,
        dst_luminance=monitor_peak_luminance
    )

    # 筆者コメント: 以下の数式は sdr_content_brightness が 80～204 あたりの場合に成立する
    #              300 や 400 を設定すると別の Tone mapping が適用されるので注意が必要
    output_hdr_contnt = modified_hdr_content * sdr_content_brightness / 203
    return output_hdr_contnt
