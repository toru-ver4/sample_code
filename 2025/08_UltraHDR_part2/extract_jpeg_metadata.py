import os
import struct


def find_markers(data: bytes, marker: bytes):
    """data 中から marker (2 バイト) のオフセット一覧を返す"""
    offs = []
    i = 0
    while True:
        i = data.find(marker, i)
        if i < 0:
            break
        offs.append(i)
        i += 2
    return offs


def parse_mpf_offsets(data: bytes):
    """
    MPF (APP2) セグメントから IFD を読んで追加画像の SOI オフセットを返す
    （ここでは 2 枚目のみを返すものとする）
    """
    APP2 = b'\xFF\xE2'
    for off in find_markers(data, APP2):
        # 長さフィールド
        seg_len = struct.unpack_from('>H', data, off+2)[0]
        payload = data[off+4 : off+2+seg_len]
        if payload.startswith(b'MPF\0\0'):
            # TIFF ヘッダ (エンディアン) は通常 MM\0* (ビッグエンディアン)
            # IFD オフセット（TIFF ヘッダからの相対）は payload[8:12]
            tiff_base = off + 4
            # entry 数は IFD オフセット直後
            ifd0_off = tiff_base + struct.unpack_from('>I', payload, 8)[0]
            num_entries = struct.unpack_from('>H', data, ifd0_off)[0]
            # エントリを順に読む
            for i in range(num_entries):
                ent_off = ifd0_off + 2 + 12*i
                tag, typ, count, val_or_off = struct.unpack_from('>HHII', data, ent_off)
                # Tag 0xB002 = MPEntry（画像情報）, 各エントリは 16 バイト × 4 個分ずれている
                if tag == 0xB002:
                    # 先頭の 4 バイトがオフセット
                    offset2 = val_or_off
                    return offset2
    raise ValueError('MPF セグメント／第2画像オフセットが見つかりません')


def extract_image_segment(data: bytes, start_off: int):
    """start_off から SOI～EOI までを切り出す"""
    soi = data.find(b'\xFF\xD8', start_off)
    if soi < 0:
        raise ValueError('SOI が見つかりません')
    eoi = data.find(b'\xFF\xD9', soi) + 2
    if eoi < 1:
        raise ValueError('EOI が見つかりません')
    return data[soi:eoi]


def extract_app1_xmp(img: bytes):
    """JPEG バイナリから APP1(XMP) を取り出す（最初に見つかったもの）"""
    APP1 = b'\xFF\xE1'
    for off in find_markers(img, APP1):
        seg_len = struct.unpack_from('>H', img, off+2)[0]
        payload = img[off+4:off+2+seg_len]
        if payload.startswith(b'http://ns.adobe.com/xap/1.0/\x00'):
            return payload
    return None


def extract_app2_iso21496(img: bytes):
    """JPEG バイナリから APP2(ISO 21496-1) を取り出す（最初に見つかったもの）"""
    APP2 = b'\xFF\xE2'
    for off in find_markers(img, APP2):
        seg_len = struct.unpack_from('>H', img, off+2)[0]
        payload = img[off+4:off+2+seg_len]
        if payload.startswith(b'ISO 21496-1'):
            return payload
    return None


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    INPUT_JPEG = './gain_map_img/HDR_Capacity_SDR_1280x720-metadata_HDR_Capacity_2.300_1280x720-HDR_Capacity_SDR_1280x720_hdr_capacity-1.414.jpg'   # ←解析対象の JPEG ファイル名
    OUT_DIR = './metadata'
    os.makedirs(OUT_DIR, exist_ok=True)

    data = open(INPUT_JPEG, 'rb').read()

    # 1) MPF から第2画像オフセットを取得
    off2 = parse_mpf_offsets(data)
    print(f'第2画像オフセット: {off2}')

    # 2) 第2画像バイナリを切り出し・保存
    img2 = extract_image_segment(data, off2)
    with open(os.path.join(OUT_DIR, 'second.jpg'), 'wb') as f:
        f.write(img2)
    print('second.jpg を保存しました')

    # 3) APP1(XMP) 抽出・保存
    xmp = extract_app1_xmp(img2)
    if xmp:
        with open(os.path.join(OUT_DIR, 'second_xmp.xml'), 'wb') as f:
            f.write(xmp)
        print('second_xmp.xml を保存しました')
    else:
        print('XMP セグメントが見つかりませんでした')

    # 4) APP2(ISO21496-1) 抽出・保存
    iso = extract_app2_iso21496(img2)
    if iso:
        with open(os.path.join(OUT_DIR, 'second_iso21496.bin'), 'wb') as f:
            f.write(iso)
        print('second_iso21496.bin を保存しました')
    else:
        print('ISO21496-1 セグメントが見つかりませんでした')
