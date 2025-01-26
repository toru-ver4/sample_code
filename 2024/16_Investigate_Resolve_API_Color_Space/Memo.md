# Fusion に関するメモ

## ダミーデータの作成

```powershell
ffmpeg.exe -f lavfi -i color=black:s=1920x1080:d=60:rate=60 -crf 35 black_60fps_1min.mp4
```

## dir(fusion) の結果

```python
>>> fusion_item = timeline.InsertFusionCompositionIntoTimeline()
>>> print(f"fusion_item = {fusion_item}")
>>> fusion_comp = fusion_item.GetFusionCompByIndex(1)
>>> pprint(dir(fusion_comp))
['AbortRender',
 'AbortRenderUI',
 'ActiveTool',
 'AddMedia',
 'AddSettingAction',
 'AddTool',
 'AddToolAction',
 'AskRenderSettings',
 'AskUser',
 'AutoPos',
 'ChooseAction',
 'ChooseTool',
 'ClearUndo',
 'Close',
 'Comp',
 'Composition',
 'Copy',
 'CopySettings',
 'CurrentFrame',
 'CurrentTime',
 'DisableSelectedTools',
 'DoAction',
 'EndUndo',
 'Execute',
 'ExpandZone',
 'Export',
 'FindTool',
 'FindToolByID',
 'GetCompPathMap',
 'GetConsoleHistory',
 'GetData',
 'GetData',
 'GetFrameList',
 'GetID',
 'GetMarkers',
 'GetNextKeyTime',
 'GetPrefs',
 'GetPrevKeyTime',
 'GetPreviewList',
 'GetRedoStack',
 'GetReg',
 'GetToolList',
 'GetUndoStack',
 'GetViewList',
 'Heartbeat',
 'IsLocked',
 'IsPlaying',
 'IsReadOnly',
 'IsRendering',
 'IsViewShowing',
 'IsZoneExpanded',
 'Lock',
 'Loop',
 'MapPath',
 'MapPathSegments',
 'NetRenderAbort',
 'NetRenderEnd',
 'NetRenderStart',
 'NetRenderTime',
 'Paste',
 'Play',
 'Print',
 'QueueAction',
 'Redo',
 'Render',
 'Reset',
 'ReverseMapPath',
 'RunScript',
 'Save',
 'SaveAs',
 'SaveCopyAs',
 'SaveVersion',
 'SetActiveTool',
 'SetData',
 'SetData',
 'SetMarker',
 'SetPrefs',
 'SetReadOnly',
 'ShowView',
 'StartUndo',
 'Stop',
 'TriggerEvent',
 'Undo',
 'Unlock',
 'UpdateMode',
 'UpdateViews',
 'XPos',
 'YPos',
 '_Output_Error',
 '_Output_Print',
 '_Save',
 '_SaveAs',
 '_SaveCopyAs',
 '_SetCurrentTime']
```

## dir(media_out) の結果

```python
>>> fusion_item = timeline.InsertFusionCompositionIntoTimeline()
>>> print(f"fusion_item = {fusion_item}")
>>> fusion_comp = fusion_item.GetFusionCompByIndex(1)
>>> media_out = fusion_comp.GetToolList()[1]
>>> pprint(dir(media_out))
['AddModifier',
 'Comp',
 'Composition',
 'Composition',
 'ConnectInput',
 'Delete',
 'DoAction',
 'FillColor',
 'FindMainInput',
 'FindMainOutput',
 'GetChildrenList',
 'GetControlPageNames',
 'GetCurrentSettings',
 'GetData',
 'GetData',
 'GetID',
 'GetInput',
 'GetInputList',
 'GetKeyFrames',
 'GetMarkers',
 'GetOutputList',
 'GetReg',
 'ID',
 'LoadSettings',
 'Name',
 'ParentTool',
 'QueueAction',
 'Refresh',
 'ResetEnabledRegion',
 'SaveSettings',
 'SetCurrentSettings',
 'SetData',
 'SetData',
 'SetInput',
 'SetMarker',
 'ShowControlPage',
 'TextColor',
 'TileColor',
 'TriggerEvent',
 'UserControls']
```

## GetInputList() の実行結果

```python
>>> fusion_item = timeline.InsertFusionCompositionIntoTimeline()
>>> print(f"fusion_item = {fusion_item}")
>>> fusion_comp = fusion_item.GetFusionCompByIndex(1)
>>> print(f"fusion_comp = {fusion_comp}")
>>> bg1 = fusion_comp.AddTool("Background")
>>> for key, value in bg1.GetInputList().items():
>>>     print(f"{key}, {value.Name}, {value.ID}")
1, Settings, SettingsNest
2, Apply Mask Inverted, ApplyMaskInverted
3, Multiply by Mask, MultiplyByMask
4, Fit Mask, FitMask
5, , Blank2
6, Channel, MaskChannel
7, Low, MaskLow
8, High, MaskHigh
9, Clip Black, MaskClipBlack
10, Clip White, MaskClipWhite
11, , Blank5
12, Motion Blur, MotionBlur
13, Quality, Quality
14, Shutter Angle, ShutterAngle
15, Center Bias, CenterBias
16, Sample Spread, SampleSpread
17, , Blank6
18, Use GPU, UseGPU
19, Hide Incoming Connections, HideInputs
20, Global In, GlobalIn
21, Global Out, GlobalOut
22, Process Mode, ProcessMode
23, Image, ImageNest
24, Width, Width
25, Height, Height
26, Pixel Aspect, PixelAspect
27, Auto Resolution, UseFrameFormatSettings
28, Depth, Depth
29, Source Color Space, Gamut.ColorSpaceNest
30, Color Space Type, Gamut.ColorType
31, Color Space, Gamut.TempColorSpace
32, Color Space, Gamut.ColorSpace
33, Red Primary, Gamut.CustomRed
34, Green Primary, Gamut.CustomGreen
35, Blue Primary, Gamut.CustomBlue
36, White Point, Gamut.CustomWhite
37, Source Gamma Space, Gamut.GammaSpaceNest
38, Curve Type, Gamut.GammaType
39, Gamma Space, Gamut.TempGammaSpace
40, Gamma Space, Gamut.GammaSpace
41, Gamma, Gamut.CustomGamma
42, Linear Limit, Gamut.CustomLimit
43, Linear Slope, Gamut.CustomSlope
44, Log Type, Gamut.TempLogType
45, Log Type, Gamut.LogType
46, Lock R/G/B, Gamut.LockRGB
47, Black, Gamut.RedBlackLevel
48, White, Gamut.RedWhiteLevel
49, Soft Clip (Knee), Gamut.RedSoftClipKnee
50, LAD, Gamut.RedLAD
51, Mid Value, Gamut.RedMidValue
52, Film Stock Gamma, Gamut.RedFilmStockGamma
53, Conversion Gamma, Gamut.RedConversionGamma
54, Black, Gamut.GreenBlackLevel
55, White, Gamut.GreenWhiteLevel
56, Green Soft Clip (Knee), Gamut.GreenSoftClipKnee
57, Green LAD, Gamut.GreenLAD
58, Green Mid Value, Gamut.GreenMidValue
59, Green Film Stock Gamma, Gamut.GreenFilmStockGamma
60, Green Conversion Gamma, Gamut.GreenConversionGamma
61, Black, Gamut.BlueBlackLevel
62, White, Gamut.BlueWhiteLevel
63, Blue Soft Clip (Knee), Gamut.BlueSoftClipKnee
64, Blue LAD, Gamut.BlueLAD
65, Blue Mid Value, Gamut.BlueMidValue
66, Blue Film Stock Gamma, Gamut.BlueFilmStockGamma
67, Blue Conversion Gamma, Gamut.BlueConversionGamma
68, Conversion Table, Gamut.ConversionTable
69, ARRI Log Version, Gamut.ARRIVersion
70, ISO / ASA / EI, Gamut.ISO
71, Film Version, Gamut.BMDFilm
72, ISO, Gamut.BMDISO
73, S-Log Version, Gamut.SLogVersion
74, Remove Curve, Gamut.GammaAction
75, Pre-Divide / Post-Multiply, Gamut.PreDividePostMultiply
76, Type, Type
77, Background, BackgroundNest
78, Top Left Red, TopLeftRed
79, Top Left Green, TopLeftGreen
80, Top Left Blue, TopLeftBlue
81, Top Left Alpha, TopLeftAlpha
82, , TopRightSep
83, Top Right Red, TopRightRed
84, Top Right Green, TopRightGreen
85, Top Right Blue, TopRightBlue
86, Top Right Alpha, TopRightAlpha
87, , BottomLeftSep
88, Bottom Left Red, BottomLeftRed
89, Bottom Left Green, BottomLeftGreen
90, Bottom Left Blue, BottomLeftBlue
91, Bottom Left Alpha, BottomLeftAlpha
92, , BottomRightSep
93, Bottom Right Red, BottomRightRed
94, Bottom Right Green, BottomRightGreen
95, Bottom Right Blue, BottomRightBlue
96, Bottom Right Alpha, BottomRightAlpha
97, Gradient Type, GradientType
98, Start, Start
99, End, End
100, Gradient, Gradient
101, Interpolation Space, GradientInterpolationMethod
102, Offset, Offset
103, Repeat, Repeat
104, Sub-Pixel, SubPixel
105, Comments, CommentsNest
106, Comments, Comments
107, Frame Render Script, FrameRenderScriptNest
108, Frame Render Script, FrameRenderScript
109, Start Render Scripts, StartRenderScripts
110, Start Render Script, StartRenderScript
111, End Render Scripts, EndRenderScripts
112, End Render Script, EndRenderScript
113, Effect Mask, EffectMask
```
