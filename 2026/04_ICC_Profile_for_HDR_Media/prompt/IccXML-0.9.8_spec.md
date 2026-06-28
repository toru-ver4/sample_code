# iccFromXml / IccXML 入力仕様メモ

この文書は、`temporary/2026/DaVinci_Photo_page/iccDEV` の `iccFromXml` が受け付ける ICC XML 形式を、今後 `ty_lib/icc_profile_calc_param.py` と `ty_lib/icc_profile_xml_control.py` を拡張するときの参照用にまとめたものです。

確認対象:

- ソース: `temporary/2026/DaVinci_Photo_page/iccDEV/IccXML`
- 主な実装: `IccProfileXml.cpp`, `IccTagXml.cpp`, `IccMpeXml.cpp`, `IccTagXmlFactory.cpp`, `IccMpeXmlFactory.cpp`
- 実行バイナリ: `takuver4/ty_env_v2:rev11` 内の `iccFromXml`
- 確認した実行時表示: `IccProfLib Version 2.3.2.1+2795b30`, `IccLibXML Version 2.3.2.1+2795b30`

ファイル名は既存運用に合わせて `IccXML-0.9.8_spec.md` としているが、上記コンテナで確認できるバイナリの表示バージョンは `2.3.2.1+2795b30` である。

## 1. コマンド仕様

基本形:

```sh
iccFromXml input.xml output.icc {-noid -v[=relax_ng_schema_file]}
```

Docker 実行例:

```sh
docker run --rm \
  -v /mnt/c/Users/toruv/OneDrive/work/sample_code:/work/src \
  -w /work/src \
  takuver4/ty_env_v2:rev12 \
  iccFromXml input.xml output.icc
```

オプション:

| オプション | 意味 |
| --- | --- |
| `-noid` | ICC 保存時に Profile ID を書き込まない。 |
| `-v` | RELAX NG 検証を有効化する。スキーマ名を省略した場合は、実行ファイル付近の `SampleIccRELAX.rng` を探す。 |
| `-v=path/to/schema.rng` | 指定した RELAX NG スキーマで検証する。 |

動作:

- XML のパースに失敗すると `Unable to Parse` で終了し、ICC は保存されない。
- XML パースには成功したが ICC Profile としての `Validate()` 結果がエラー以上でも、保存可能なら ICC は保存される。その場合は「Profile is invalid, but saved correctly」と検証レポートが出る。
- `ProfileID` が XML にあり、かつ `-noid` がない場合は ID を保存する。ID が空の場合は ICC バージョンに応じた扱いになる。
- CLI 版では `<TextData File="...">` や `<Data File="...">` などのファイル include が許可される。ライブラリ呼び出しでは既定で無効にされる設計。
- libxml2 のネットワークアクセスは無効化されている。外部ネットワーク Entity を使う前提にはしない。

## 2. XML 全体構造

最小のトップレベル構造:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<IccProfile>
  <Header>
    ...
  </Header>
  <Tags>
    ...
  </Tags>
</IccProfile>
```

制約:

- ルート要素は `IccProfile` 固定。
- `Header` と `Tags` は必須。
- `Header` は先に解釈される。MBB/MPE 系タグの色空間設定、Named Color の PCS/Data 色空間設定は Header の値に依存する。
- 不明な Header 要素はパースエラーにはならず、警告文字列に「Unknown Profile Header attribute」として蓄積される。
- `Tags` 直下の各要素は ICC tag 1 個を表す。tag 名ラッパー形式と legacy tag type 直書き形式の 2 系統を受け付ける。

## 3. Header

`Header` の子要素:

| 要素 | 内容 |
| --- | --- |
| `PreferredCMMType` | 4 文字 signature。空文字可。 |
| `ProfileVersion` | ICC バージョン。例: `4.30`, `5.10`。`.` または `,` 区切り。 |
| `ProfileSubClassVersion` | v5 サブクラスバージョン。省略可。 |
| `ProfileDeviceClass` | 4 文字 profile class signature。例: `mntr`, `spac`, `link`, `nmcl`。 |
| `ProfileDeviceSubClass` | 4 文字 subclass signature。省略可。 |
| `DataColourSpace` | 4 文字 color space signature。例: `RGB `, `XYZ `, `Lab `, `MCS `。末尾スペースに注意。 |
| `PCS` | 4 文字 PCS signature。例: `XYZ `, `Lab `, `Jab `。 |
| `CreationDateTime` | `YYYY-MM-DDTHH:MM:SS` または `now`。 |
| `PrimaryPlatform` | platform signature。省略可。 |
| `ProfileFlags` | 属性で指定する。 |
| `DeviceManufacturer` | 4 文字 signature。省略可。 |
| `DeviceModel` | 4 文字 signature。省略可。 |
| `DeviceAttributes` | 属性で指定する。 |
| `RenderingIntent` | `Perceptual`, `Relative Colorimetric`, `Relative`, `Saturation`, `Absolute Colorimetric`, `Absolute`。 |
| `PCSIlluminant` | 子に `XYZNumber` を持つ。 |
| `ProfileCreator` | 4 文字 signature。空文字可。 |
| `ProfileID` | 16 byte hex。短い値も読み込むが、生成側は 32 桁 hex を推奨。 |
| `SpectralPCS` | spectral PCS color signature。v5/スペクトル系で使用。 |
| `SpectralRange` | 子に `Wavelengths start="..." end="..." steps="..."`。 |
| `BiSpectralRange` | 子に `Wavelengths start="..." end="..." steps="..."`。 |
| `MCS` | multiplex color signature。 |
| `Reserved` | reserved byte列を hex で指定。 |

`ProfileFlags`:

```xml
<ProfileFlags
  EmbeddedInFile="true"
  UseWithEmbeddedDataOnly="false"
  ExtendedRangePCS="true"
  MCSNeedsSubset="false"
  VendorFlags="00000000"/>
```

実装で `true` と比較しているため、真値は小文字 `true` を使う。`VendorFlags` は hex として OR される。

`DeviceAttributes`:

```xml
<DeviceAttributes
  ReflectiveOrTransparency="reflective"
  GlossyOrMatte="glossy"
  MediaPolarity="positive"
  MediaColour="colour"/>
```

`PCSIlluminant`:

```xml
<PCSIlluminant>
  <XYZNumber X="0.9642" Y="1.0000" Z="0.8249"/>
</PCSIlluminant>
```

`SpectralRange` / `BiSpectralRange`:

```xml
<SpectralRange>
  <Wavelengths start="400" end="700" steps="31"/>
</SpectralRange>
```

## 4. Tags の指定方式

### 4.1 推奨: tag 名ラッパー形式

既知の ICC tag 名をラッパーにし、その最初の子要素として tag type を置く。

```xml
<profileDescriptionTag>
  <multiLocalizedUnicodeType>
    <LocalizedText LanguageCountry="enUS"><![CDATA[Display profile]]></LocalizedText>
  </multiLocalizedUnicodeType>
</profileDescriptionTag>
```

既知 tag 名は `CIccInfo::GetTagSigName()` / `icGetTagNameSig()` で引ける名前で、`iccToXml` が出力する `profileDescriptionTag`, `copyrightTag`, `mediaWhitePointTag`, `AToB0Tag`, `BToA1Tag`, `cicpTag`, `spectralViewingConditionsTag` などをそのまま使える。

### 4.2 独自 tag: `PrivateTag`

未知 tag signature は `PrivateTag` に `TagSignature` を付ける。

```xml
<PrivateTag TagSignature="MHC2">
  <PrivateType type="MHC2">
    <UnknownData>
      00 01 02 03
    </UnknownData>
  </PrivateType>
</PrivateTag>
```

`PrivateType`:

- `type="...."` で 4 byte tag type signature を指定する。
- 子には通常 `UnknownData` hex を置く。
- `reserved="00000000"` 属性を付けると tag type の reserved field に反映される。

### 4.3 SameAs

複数 tag が同じ tag object / offset を共有する場合に使う。

```xml
<greenTRCTag SameAs="redTRCTag"/>
<blueTRCTag SameAs="redTRCTag"/>
```

PrivateTag を参照する場合:

```xml
<PrivateTag TagSignature="abcd" SameAs="PrivateTag" SameAsSignature="MHC2"/>
```

参照先 tag は XML 上で先に出現している必要がある。

### 4.4 legacy: tag type 直書き形式

古い XML では `Tags` 直下に tag type 要素を直接置き、中に 1 個以上の `TagSignature` を置く。

```xml
<curveType>
  <TagSignature>rTRC</TagSignature>
  <TagSignature>gTRC</TagSignature>
  <TagSignature>bTRC</TagSignature>
  <Curve>...</Curve>
</curveType>
```

現在の生成コードでは tag 名ラッパー形式を推奨する。legacy 形式は互換用として認識する。

## 5. 共通データ表現

### 5.1 signature

ICC signature は 4 byte 文字列として扱う。`RGB `, `XYZ ` のように末尾スペースが有意な場合がある。Python 側では trim しないこと。

### 5.2 数値配列

多くの配列は空白区切りテキストを受け付ける。

```xml
<Array>
  1.0 0.0 0.0
  0.0 1.0 0.0
  0.0 0.0 1.0
</Array>
```

互換構文として整数配列は `<n>...</n>`、float 配列は `<f>...</f>` の繰り返しも受け付ける。ただし生成側は読みやすい空白区切りを推奨する。

数値文字として認識されるのは数字、`.`, `+`, `-`, `e`, `n`, `a`。float では `nan` / `-nan` を扱える。整数型へ NaN を入れた場合は 0 に丸められる。

### 5.3 hex data

`UnknownData`, `Data`, `HexTextData`, `HexCompressedData`, `Reserved`, `ProfileID` などは hex 文字列を読む。空白と改行を含められる。

### 5.4 text data

テキスト型は主に以下を受け付ける。

```xml
<TextData><![CDATA[text]]></TextData>
<TextData File="relative/or/absolute/path.txt"/>
<HexTextData>48 65 6C 6C 6F</HexTextData>
```

`File` 属性は CLI 版では有効。安全なパス判定に失敗すると読めない。テキストは XML 側では UTF-8 として扱い、ICC の型に応じて ANSI/UTF-16 等に変換される。

### 5.5 localized text

`multiLocalizedUnicodeType` では 1 個以上の `LocalizedText` が必要。

```xml
<multiLocalizedUnicodeType>
  <LocalizedText LanguageCountry="enUS"><![CDATA[Profile description]]></LocalizedText>
  <LocalizedText LanguageCountry="jaJP"><![CDATA[プロファイル説明]]></LocalizedText>
</multiLocalizedUnicodeType>
```

属性名は `LanguageCountry`, `languageCountry`, `LanguangeCountry` を読むが、生成側は `LanguageCountry` に統一する。

## 6. サポート tag type 一覧

`IccTagXmlFactory.cpp` で XML extension が用意される tag type:

| XML tag type | 主な中身 |
| --- | --- |
| `signatureType` | `<Signature>....</Signature>` |
| `textType` | `TextData` / `HexTextData` |
| `utf8Type` | UTF-8 `TextData` / `HexTextData` |
| `utf16Type` | UTF-16 用テキスト。XML 側は UTF-8 text として指定。 |
| `zipUtf8Type` | `HexCompressedData` または非圧縮 text から生成。 |
| `zipXmlType` | `HexCompressedData` または非圧縮 XML text から生成。 |
| `textDescriptionType` | v2 desc 互換。`TextData`, 任意で `Unicode`, `MacScript`。 |
| `XYZArrayType` | 1 個以上の `XYZNumber X="..." Y="..." Z="..."`。 |
| `cicpType` | `cicpFields ColorPrimaries="..." TransferCharacteristics="..." MatrixCoefficients="..." VideoFullRangeFlag="..."`。 |
| `uint8ArrayType` | `Array` または `Data` の整数列。 |
| `uint16ArrayType` | `Array` または `Data` の整数列。 |
| `uint32ArrayType` | `Array` または `Data` の整数列。 |
| `uint64ArrayType` | `Array` または `Data` の整数列。 |
| `s15Fixed16ArrayType` | `Array` の float 列。s15Fixed16 に変換。 |
| `u16Fixed16ArrayType` | `Array` の float 列。u16Fixed16 に変換。 |
| `float16ArrayType` | `Data` の float 列。`File` / `Filename` 読込可。 |
| `float32ArrayType` | `Data` の float 列。`File` / `Filename` 読込可。 |
| `float64ArrayType` | `Data` の float 列。`File` / `Filename` 読込可。 |
| `gamutBoundaryDescType` | `Vertices` と `Triangles`。 |
| `curveType` | `Curve`。 |
| `segmentedCurveType` | `SegmentedCurve`。 |
| `parametricCurveType` | `ParametricCurve`。 |
| `measurementType` | `StandardObserver`, `MeasurementBacking`, `Geometry`, `Flare`, `StandardIlluminant`。 |
| `multiLocalizedUnicodeType` | `LocalizedText` の繰り返し。 |
| `multiProcessElementType` | `MultiProcessElements` と MPE 要素列。 |
| `lutAtoBType` | MBB 構造。A/B/M/CLUT/Matrix/Curve を順序指定。 |
| `lutBtoAType` | MBB 構造。 |
| `lut16Type` | mft2 LUT。 |
| `lut8Type` | mft1 LUT。 |
| `namedColor2Type` | `NamedColors`。 |
| `chromaticityType` | `Colorant` と `Channel x="..." y="..."`。 |
| `dataType` | `Data` hex。 |
| `dateTimeType` | `DateTime`。 |
| `colorantOrderType` | `ColorantOrder` と `<n>` values。 |
| `colorantTableType` | `ColorantTable` と `Colorant` entries。 |
| `sparseMatrixArrayType` | `SparseMatrixArray`。 |
| `viewingConditionsType` | `IlluminantXYZ`, `SurroundXYZ`, `IllumType`。 |
| `spectralViewingConditionsType` | `StdObserver`, `IlluminantXYZ`, `ObserverFuncs`, `StdIlluminant`, `ColorTemperature`, `IlluminantSPD`, `SurroundXYZ`。 |
| `spectralDataInfoType` | `SpectralSpace`, `SpectralRange`, 任意の `BiSpectralRange`。 |
| `profileSequenceDescType` | `ProfileSequence`。 |
| `responseCurveSet16Type` | `CountOfChannels`, `ChannelResponses`。 |
| `profileSequenceIdentifierType` | `ProfileSequenceId`, `ProfileIdDesc`。 |
| `dictType` | `DictEntry` の繰り返し。 |
| `tagStructType` | `StructureSignature`, `MemberTags`。 |
| `tagArrayType` | `ArraySignature`, `ArrayTags`。 |
| `embeddedProfileType` | 子に `IccProfile`。 |
| `embeddedHeightImageType` | `HeightImage`。 |
| `embeddedNormalImageType` | `NormalImage`。 |
| `PrivateType` | `type="...."` と `UnknownData`。未知 tag type 用。 |

`screeningType`, `ucrBgType`, `crdInfoType` は factory の switch に現れるが XML 専用実装はなく、未知型扱いになる。

## 7. 主要 tag type の詳細

### 7.1 `XYZArrayType`

```xml
<mediaWhitePointTag>
  <XYZArrayType>
    <XYZNumber X="0.9642" Y="1.0000" Z="0.8249"/>
  </XYZArrayType>
</mediaWhitePointTag>
```

`XYZNumber` は 1 個以上。内部では s15Fixed16 XYZ に変換される。

### 7.2 `cicpType`

```xml
<cicpTag>
  <cicpType>
    <cicpFields ColorPrimaries="9"
                TransferCharacteristics="16"
                MatrixCoefficients="0"
                VideoFullRangeFlag="1"/>
  </cicpType>
</cicpTag>
```

属性がない場合は 0 になる。HDR/BT.2100 系では `ColorPrimaries=9`, `TransferCharacteristics=16(PQ)` または `18(HLG)` などを使う。

### 7.3 `multiLocalizedUnicodeType`

```xml
<profileDescriptionTag>
  <multiLocalizedUnicodeType>
    <LocalizedText LanguageCountry="enUS"><![CDATA[BT.2100 PQ Full Range]]></LocalizedText>
  </multiLocalizedUnicodeType>
</profileDescriptionTag>
```

少なくとも 1 個の `LocalizedText` が必要。

### 7.4 `curveType`

```xml
<redTRCTag>
  <curveType>
    <Curve>0 1024 2048 4095</Curve>
  </curveType>
</redTRCTag>
```

`Curve` の内容は実装上 `CIccTagXmlCurve::ParseXml` が読む。要素数 0/1/複数で ICC curveType の意味が変わるため、生成側は狙う表現を明確にする。

### 7.5 `parametricCurveType`

```xml
<redTRCTag>
  <parametricCurveType>
    <ParametricCurve FunctionType="3">2.4 1.0 0.0 0.0 0.0</ParametricCurve>
  </parametricCurveType>
</redTRCTag>
```

`FunctionType` は ICC parametric curve の関数番号。本文はその関数に必要なパラメータ列。

### 7.6 `segmentedCurveType`

```xml
<segmentedCurveType>
  <SegmentedCurve>
    <FormulaSegment Start="-infinity" End="0.0" FunctionType="0">1 0 0 0</FormulaSegment>
    <FormulaSegment Start="0.0" End="+infinity" FunctionType="6">...</FormulaSegment>
  </SegmentedCurve>
</segmentedCurveType>
```

`SegmentedCurve` 内では `FormulaSegment` と `SampledSegment` を扱う。MPE の `CurveSetElement` 内でも同じ curve parser が使われる。

### 7.7 `namedColor2Type`

```xml
<namedColor2Tag>
  <namedColor2Type>
    <NamedColors VendorFlag="00000000"
                 CountOfDeviceCoords="3"
                 DeviceEncoding="int16"
                 Prefix=""
                 Suffix="">
      <XYZNamedColor Name="white" X="0.9642" Y="1.0" Z="0.8249">65535 65535 65535</XYZNamedColor>
      <LabNamedColor Name="gray" L="50" a="0" b="0">32768 32768 32768</LabNamedColor>
    </NamedColors>
  </namedColor2Type>
</namedColor2Tag>
```

`DeviceEncoding` は `int8`, `int16`, `float`。`NamedColor`, `LabNamedColor`, `XYZNamedColor` を読む。

### 7.8 `spectralViewingConditionsType`

```xml
<spectralViewingConditionsTag>
  <spectralViewingConditionsType>
    <StdObserver>CIE 1931 (two degree) standard observer</StdObserver>
    <IlluminantXYZ X="0.9642" Y="1.0" Z="0.8249"/>
    <StdIlluminant>Illuminant D50</StdIlluminant>
    <ColorTemperature>5000</ColorTemperature>
    <SurroundXYZ X="0.2" Y="0.2" Z="0.2"/>
  </spectralViewingConditionsType>
</spectralViewingConditionsTag>
```

`ObserverFuncs` や `IlluminantSPD` を使う場合は、子に数値配列を置く。サンプルでは abbreviated な `StdObserver`/`StdIlluminant` 形式が多い。

### 7.9 `tagStructType`

```xml
<PrivateTag TagSignature="xxxx">
  <tagStructType>
    <StructureSignature>abcd</StructureSignature>
    <MemberTags>
      <PrivateTag TagSignature="m001">
        <utf8Type>
          <TextData><![CDATA[value]]></TextData>
        </utf8Type>
      </PrivateTag>
    </MemberTags>
  </tagStructType>
</PrivateTag>
```

既知 structure signature の場合は、structure 名を要素名として使う形も `iccToXml` 出力に現れる。未知構造では `StructureSignature` を明示する。

### 7.10 `tagArrayType`

```xml
<PrivateTag TagSignature="xxxx">
  <tagArrayType>
    <ArraySignature>abcd</ArraySignature>
    <ArrayTags>
      <utf8Type>
        <TextData><![CDATA[item0]]></TextData>
      </utf8Type>
      <utf8Type>
        <TextData><![CDATA[item1]]></TextData>
      </utf8Type>
    </ArrayTags>
  </tagArrayType>
</PrivateTag>
```

array 要素もそれぞれ通常の tag type と同じ XML parser で解釈される。

## 8. MPE: `multiProcessElementType`

基本形:

```xml
<AToB1Tag>
  <multiProcessElementType>
    <MultiProcessElements InputChannels="3" OutputChannels="3">
      <CurveSetElement InputChannels="3" OutputChannels="3">...</CurveSetElement>
      <MatrixElement InputChannels="3" OutputChannels="3">...</MatrixElement>
      <CLutElement InputChannels="3" OutputChannels="3">...</CLutElement>
    </MultiProcessElements>
  </multiProcessElementType>
</AToB1Tag>
```

`MultiProcessElements` には profile 全体の input/output channel 数を属性で指定する。各 element も `InputChannels` と `OutputChannels` を持つ。

XML extension がある MPE 要素:

| MPE 要素 | 内容 |
| --- | --- |
| `MatrixElement` | `MatrixData`, 任意で `ConstantData` または `OffsetData`。 |
| `CurveSetElement` | channel ごとの curve。`SegmentedCurve`, `SampledCurve`, `CalculatorCurve`, `SingleSampledCurve`, `DuplicateCurve` など。 |
| `CLutElement` | `GridPoints`, `TableData`。 |
| `ExtCLutElement` | 拡張 CLUT。 |
| `CalculatorElement` | `Variables`, `Macros`, `SubElements`, `MainFunction`。 |
| `TintArrayElement` | 内部に tag array 相当の型を持つ。 |
| `ToneMapElement` | `LuminanceCurve`, `ToneMapFunctions`。 |
| `XYZToJabElement` | `ColorAppearanceParams`。 |
| `JabToXYZElement` | `ColorAppearanceParams`。 |
| `EmissionMatrixElement` | `Wavelengths`, `WhiteData`, `MatrixData`, `OffsetData`。 |
| `InvEmissionMatrixElement` | `Wavelengths`, `WhiteData`, `MatrixData`, `OffsetData`。 |
| `EmissionCLutElement` | spectral CLUT。 |
| `ReflectanceCLutElement` | spectral CLUT。 |
| `EmissionObserverElement` | observer data。 |
| `ReflectanceObserverElement` | observer data。 |
| `BAcsElement` | begin ACS marker。 |
| `EAcsElement` | end ACS marker。 |

### 8.1 `MatrixElement`

```xml
<MatrixElement InputChannels="3" OutputChannels="3">
  <MatrixData InvertMatrix="true">
    0.63695805 0.14461690 0.16888098
    0.26270021 0.67799807 0.05930172
    0.00000000 0.02807269 1.06098506
  </MatrixData>
</MatrixElement>
```

- `MatrixData` は `OutputChannels * InputChannels` 個の float。
- `InvertMatrix="true"` を付けると実装側で逆行列化して格納する。
- `ConstantData` または旧名 `OffsetData` は output channel 数分の offset。

### 8.2 `CurveSetElement`

```xml
<CurveSetElement InputChannels="3" OutputChannels="3">
  <SegmentedCurve>
    <FormulaSegment Start="-infinity" End="0.0" FunctionType="0">1 0 0 0</FormulaSegment>
    <FormulaSegment Start="0" End="+infinity" FunctionType="6">...</FormulaSegment>
  </SegmentedCurve>
  <DuplicateCurve Index="0"/>
  <DuplicateCurve Index="0"/>
</CurveSetElement>
```

- curve は output channel 数分必要。
- `DuplicateCurve Index="0"` は既出 curve を複製する。
- `FormulaSegment` の `Start` / `End` は `-infinity`, `+infinity` を受け付ける。
- `FunctionType` は formula curve segment の関数番号。本文はパラメータ列。

### 8.3 `CLutElement`

```xml
<CLutElement InputChannels="3" OutputChannels="3">
  <GridPoints>17 17 17</GridPoints>
  <TableData>
    ...
  </TableData>
</CLutElement>
```

`GridPoints` は input channel 数分、`TableData` は grid 総点数 * output channel 数分の float。

### 8.4 `ToneMapElement`

```xml
<ToneMapElement InputChannels="4" OutputChannels="3">
  <LuminanceCurve>
    <SegmentedCurve>
      <FormulaSegment Start="0" End="1" FunctionType="0">...</FormulaSegment>
    </SegmentedCurve>
  </LuminanceCurve>
  <ToneMapFunctions>
    <ToneMapFunction FunctionType="0">1 0 0</ToneMapFunction>
    <DuplicateFunction Index="0"/>
    <DuplicateFunction Index="0"/>
  </ToneMapFunctions>
</ToneMapElement>
```

`LuminanceCurve` は curve parser を使う。`ToneMapFunctions` は `ToneMapFunction` と `DuplicateFunction` を使う。

### 8.5 `XYZToJabElement` / `JabToXYZElement`

```xml
<XYZToJabElement InputChannels="3" OutputChannels="3">
  <ColorAppearanceParams>
    <WhitePoint><XYZNumber X="0.9505" Y="1.0" Z="1.0890"/></WhitePoint>
    <Luminance>100</Luminance>
    <BackgroundLuminance>20</BackgroundLuminance>
    <ImpactSurround>1.0</ImpactSurround>
    <ChromaticInductionFactor>1.0</ChromaticInductionFactor>
    <AdaptationFactor>1.0</AdaptationFactor>
  </ColorAppearanceParams>
</XYZToJabElement>
```

`ColorAppearanceParams` は `WhitePoint`, `Luminance`, `BackgroundLuminance`, `ImpactSurround`, `ChromaticInductionFactor`, `AdaptationFactor` を読む。

### 8.6 `CalculatorElement`

主な構造:

```xml
<CalculatorElement InputChannels="3" OutputChannels="3">
  <Variables>
    <Declare Name="x" StorageType="float"/>
  </Variables>
  <Macros>
    <Macro Name="m">...</Macro>
  </Macros>
  <SubElements>
    <MatrixElement InputChannels="3" OutputChannels="3">...</MatrixElement>
  </SubElements>
  <MainFunction>
    ...
  </MainFunction>
</CalculatorElement>
```

実装は import、変数宣言、macro、sub element、main function を持つ計算要素を解釈する。Calc 系サンプル (`Testing/Calc/*.xml`) はこの構文の参照として重要。

## 9. LUT/MBB 系 tag

`lutAtoBType`, `lutBtoAType`, `lut8Type`, `lut16Type` は ICC v4/v2 の LUT 構造を XML 化したもの。

実装上、`AToB0Tag` / `AToB1Tag` / `AToB2Tag` / `AToB3Tag` は Header の `DataColourSpace -> PCS`、`BToA*Tag` は `PCS -> DataColourSpace`、`HToS*Tag` は `PCS -> PCS`、`gamutTag` は `PCS -> gamut` として色空間が補われる。

新規生成では、HDR/PCC/v5 拡張の柔軟性を考えると `multiProcessElementType` を優先し、既存互換が必要な場合だけ LUT/MBB 系を使うのが扱いやすい。

## 10. 独自/未知型の扱い

未知 tag type でも、4 byte type signature と raw data が分かれば `PrivateType` + `UnknownData` で保存できる。

```xml
<PrivateTag TagSignature="ABCD">
  <PrivateType type="WXYZ" reserved="00000000">
    <UnknownData>
      00 00 00 01  12 34 56 78
    </UnknownData>
  </PrivateType>
</PrivateTag>
```

注意:

- `PrivateTag` の `TagSignature` は必須。
- `PrivateType` の `type` がないと unknown type として作成できない。
- raw byte の endian は ICC tag type の仕様に従う。`UnknownData` は byte 列をそのまま書く。

## 11. 生成側で守るべき実用ルール

- 4 文字 signature の末尾スペースを保持する。`RGB `, `XYZ `, `Lab ` は 4 文字。
- Header を先に完全に作る。特に `ProfileVersion`, `ProfileDeviceClass`, `DataColourSpace`, `PCS`, `RenderingIntent`, `PCSIlluminant`。
- tag は tag 名ラッパー形式に統一する。legacy の `TagSignature` 方式は読み込み互換用に留める。
- `LocalizedText` は `LanguageCountry="enUS"` を最低 1 つ入れる。
- 数値配列は空白区切り text に統一する。`<n>`/`<f>` は出力しない。
- float は小数点付きで十分な桁を出す。ICC fixed に変換される型では丸めが入る。
- `CreationDateTime` は再現性が必要なら固定日時、手元生成なら `now` または `YYYY-MM-DDTHH:MM:SS`。
- `ProfileID` は必要なければ空/省略し、最終保存時にバイナリ側へ任せる。固定したい場合は 16 byte = 32 hex 桁を出す。
- 独自 tag は最初から `PrivateTag` / `PrivateType` で実装し、既知 tag 名への偽装はしない。
- `SameAs` は参照先を必ず先に出力する。
- `File` / `Filename` include は実行環境依存になるため、ライブラリ生成では原則インライン text/hex/data を推奨する。

## 12. 代表テンプレート

### 12.1 RGB display profile skeleton

```xml
<?xml version="1.0" encoding="UTF-8"?>
<IccProfile>
  <Header>
    <PreferredCMMType></PreferredCMMType>
    <ProfileVersion>4.30</ProfileVersion>
    <ProfileDeviceClass>mntr</ProfileDeviceClass>
    <DataColourSpace>RGB </DataColourSpace>
    <PCS>XYZ </PCS>
    <CreationDateTime>2026-01-01T00:00:00</CreationDateTime>
    <ProfileFlags EmbeddedInFile="false" UseWithEmbeddedDataOnly="false"/>
    <DeviceAttributes ReflectiveOrTransparency="reflective" GlossyOrMatte="glossy" MediaPolarity="positive" MediaColour="colour"/>
    <RenderingIntent>Relative Colorimetric</RenderingIntent>
    <PCSIlluminant>
      <XYZNumber X="0.9642" Y="1.0000" Z="0.8249"/>
    </PCSIlluminant>
    <ProfileCreator></ProfileCreator>
  </Header>
  <Tags>
    <profileDescriptionTag>
      <multiLocalizedUnicodeType>
        <LocalizedText LanguageCountry="enUS"><![CDATA[Example RGB display profile]]></LocalizedText>
      </multiLocalizedUnicodeType>
    </profileDescriptionTag>
    <copyrightTag>
      <multiLocalizedUnicodeType>
        <LocalizedText LanguageCountry="enUS"><![CDATA[Copyright]]></LocalizedText>
      </multiLocalizedUnicodeType>
    </copyrightTag>
    <mediaWhitePointTag>
      <XYZArrayType>
        <XYZNumber X="0.9642" Y="1.0000" Z="0.8249"/>
      </XYZArrayType>
    </mediaWhitePointTag>
  </Tags>
</IccProfile>
```

### 12.2 HDR/PQ display transform skeleton

```xml
<AToB1Tag>
  <multiProcessElementType>
    <MultiProcessElements InputChannels="3" OutputChannels="3">
      <CurveSetElement InputChannels="3" OutputChannels="3">
        <SegmentedCurve>
          <FormulaSegment Start="-infinity" End="0.0" FunctionType="0">1 0 0 0</FormulaSegment>
          <FormulaSegment Start="0" End="+infinity" FunctionType="6">0.1593017578125 0.01268331351565597 0.8359375 18.8515625 18.6875 10000 1.0</FormulaSegment>
        </SegmentedCurve>
        <DuplicateCurve Index="0"/>
        <DuplicateCurve Index="0"/>
      </CurveSetElement>
      <MatrixElement InputChannels="3" OutputChannels="3">
        <MatrixData>
          0.63695805 0.14461690 0.16888098
          0.26270021 0.67799807 0.05930172
          0.00000000 0.02807269 1.06098506
        </MatrixData>
      </MatrixElement>
    </MultiProcessElements>
  </multiProcessElementType>
</AToB1Tag>
```

## 13. 参照すべき iccDEV サンプル

`iccFromXml` の構文確認には以下が有用。

| 用途 | サンプル |
| --- | --- |
| HDR PQ/HLG v5 profile | `temporary/2026/DaVinci_Photo_page/iccDEV/Testing/HDR/*.xml` |
| Rec.2100 display profile | `temporary/2026/DaVinci_Photo_page/iccDEV/Testing/Display/Rec2100HlgFull.xml` |
| MPE calculator | `temporary/2026/DaVinci_Photo_page/iccDEV/Testing/Calc/*.xml` |
| PCC / spectral profile | `temporary/2026/DaVinci_Photo_page/iccDEV/Testing/PCC/*.xml` |
| named color / tagStruct / tagArray | `temporary/2026/DaVinci_Photo_page/iccDEV/Testing/Named/*.xml` |
| spec reference | `temporary/2026/DaVinci_Photo_page/iccDEV/Testing/SpecRef/*.xml` |

## 14. Python ライブラリ拡張への示唆

今後の `icc_profile_xml_control.py` は、既存の個別タグ編集だけでなく、以下の抽象を持つと拡張しやすい。

- `IccProfile` root builder: Header と Tags の順序を固定。
- Header builder: signature の 4 文字保持、日時、flags、device attributes、spectral range を責務にする。
- Tag wrapper builder: `<knownTagName><tagType>...</tagType></knownTagName>` と `<PrivateTag TagSignature=...>` を共通化。
- Tag type builders: `multiLocalizedUnicodeType`, `XYZArrayType`, `cicpType`, `curveType`, `parametricCurveType`, `multiProcessElementType`, `PrivateType` をまず分離。
- MPE builders: `MatrixElement`, `CurveSetElement`, `CLutElement`, `ToneMapElement`, `CalculatorElement` を独立させる。
- Array/text utilities: 空白区切り数値配列、CDATA、hex dump、localized text を共通化。

特に HDR Media 用のプロファイルでは、`multiProcessElementType` と `cicpType`, `spectralViewingConditionsType`, `customToStandardPccTag`, `standardToCustomPccTag` が増えるため、既存の TRC/XYZ/chad 専用ロジックに閉じない設計にする。
