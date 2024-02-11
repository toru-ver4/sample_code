# -*- coding: utf-8 -*-
"""

"""

# import standard libraries
import os

# import third-party libraries

# import my libraries

# Since the input was truncated, I'll demonstrate the processing with a sample from the provided text
# and then apply the same logic to the entire text.

text_dump = """
audioCaptureNumChannels: 2
audioOutputHasTimecode: 0
audioPlayoutNumChannels: 2
colorAcesGamutCompressType: None
colorAcesIDT: No Input Transform
colorAcesNodeLUTProcessingSpace: acesccAp1
colorAcesODT: No Output Transform
colorGalleryStillsLocation: D:\Resolve\.gallery
colorGalleryStillsNamingCustomPattern: 
colorGalleryStillsNamingEnabled: 0
colorGalleryStillsNamingPattern: clipName
colorGalleryStillsNamingWithStillNumber: off
colorKeyframeDynamicsEndProfile: 1
colorKeyframeDynamicsStartProfile: 1
colorLuminanceMixerDefaultZero: 0
colorScienceMode: davinciYRGBColorManagedv2
colorSpaceInput: Rec.2020
colorSpaceInputGamma: ST2084
colorSpaceOutput: Rec.2020
colorSpaceOutputGamma: ST2084
colorSpaceOutputGamutMapping: None
colorSpaceOutputGamutSaturationKnee: 0.9
colorSpaceOutputGamutSaturationMax: 1
colorSpaceOutputToneLuminanceMax: 100
colorSpaceOutputToneMapping: None
colorSpaceTimeline: Rec.2020
colorSpaceTimelineGamma: ST2084
colorUseBGRPixelOrderForDPX: 0
colorUseContrastSCurve: 1
colorUseLegacyLogGrades: 2
colorUseLocalVersionsAsDefault: 1
colorUseStereoConvergenceForEffects: 0
colorVersion10Name: 
colorVersion1Name: 
colorVersion2Name: 
colorVersion3Name: 
colorVersion4Name: 
colorVersion5Name: 
colorVersion6Name: 
colorVersion7Name: 
colorVersion8Name: 
colorVersion9Name: 
disableFusionToneMapping: 0
graphicsWhiteLevel: 200
hdr10PlusControlsOn: 0
hdrDolbyAnalysisTuning: Balanced
hdrDolbyControlsOn: 0
hdrDolbyMasterDisplay: 4000-nit, P3, D65, ST.2084, Full
hdrDolbyVersion: 4.0
hdrMasteringLuminanceMax: 1000
hdrMasteringOn: 0
imageDeinterlaceQuality: normal
imageEnableFieldProcessing: 0
imageMotionEstimationMode: standardFaster
imageMotionEstimationRange: medium
imageResizeMode: sharper
imageResizingGamma: Log
imageRetimeInterpolation: nearest
inputDRT: None
inputDRTSatRolloffLimit: 10000
inputDRTSatRolloffStart: 100
isAutoColorManage: 0
limitAudioMeterAlignLevel: 0
limitAudioMeterDisplayMode: post_fader
limitAudioMeterHighLevel: -5
limitAudioMeterLUFS: -23
limitAudioMeterLoudnessScale: ebu_9_scale
limitAudioMeterLowLevel: -10
limitBroadcastSafeLevels: 20_120
limitBroadcastSafeOn: 0
limitSubtitleCPL: 60
limitSubtitleCaptionDurationSec: 3
outputDRT: None
outputDRTSatRolloffLimit: 10000
outputDRTSatRolloffStart: 100
perfAutoRenderCacheAfterTime: 5
perfAutoRenderCacheComposite: 0
perfAutoRenderCacheEnable: 1
perfAutoRenderCacheFuEffect: 1
perfAutoRenderCacheTransition: 0
perfCacheClipsLocation: D:\Resolve\CacheClip
perfOptimisedCodec: dnxhd_hqx_12b
perfOptimisedMediaOn: 1
perfOptimizedResolutionRatio: auto
perfProxyMediaMode: 1
perfProxyResolutionRatio: original
perfRenderCacheCodec: dnxhd_hqx_12b
perfRenderCacheMode: none
rcmPresetMode: Custom
separateColorSpaceAndGamma: 1
superScaleNoiseReduction: Medium
superScaleNoiseReductionStrength: 0.5
superScaleSharpness: Medium
superScaleSharpnessStrength: 0.5
timelineDropFrameTimecode: 0
timelineFrameRate: 24.0
timelineFrameRateMismatchBehavior: resolve
timelineInputResMismatchBehavior: scaleToFit
timelineInputResMismatchCustomPreset: None
timelineInputResMismatchUseCustomPreset: 0
timelineInterlaceProcessing: 0
timelineOutputPixelAspectRatio: square
timelineOutputResMatchTimelineRes: 1
timelineOutputResMismatchBehavior: scaleToFit
timelineOutputResMismatchCustomPreset: None
timelineOutputResMismatchUseCustomPreset: 0
timelineOutputResolutionHeight: 1080
timelineOutputResolutionWidth: 1920
timelinePixelAspectRatio: square
timelinePlaybackFrameRate: 24
timelineResolutionHeight: 1080
timelineResolutionWidth: 1920
timelineSaveThumbsInProject: 0
timelineWorkingLuminance: 10000
timelineWorkingLuminanceMode: Custom
transcriptionLanguage: auto
useCATransform: 1
useColorSpaceAwareGradingTools: 1
useInverseDRT: 1
videoCaptureCodec: rgb
videoCaptureFormat: dpx
videoCaptureIngestHandles: 0
videoCaptureLocation: D:\Resolve\Capture
videoCaptureMode: video_audio
videoDataLevels: Video
videoDataLevelsRetainSubblockAndSuperWhiteData: 0
videoDeckAdd32Pulldown: 0
videoDeckBitDepth: 10
videoDeckFormat: HD 1080PsF 24
videoDeckNonAutoEditFrames: 0
videoDeckOutputSyncSource: auto
videoDeckPrerollSec: 5
videoDeckSDIConfiguration: none
videoDeckUse444SDI: 0
videoDeckUseAudoEdit: 1
videoDeckUseStereoSDI: 0
videoMonitorBitDepth: 10
videoMonitorFormat: HD 1080p 24
videoMonitorMatrixOverrideFor422SDI: Rec.601
videoMonitorSDIConfiguration: single_link
videoMonitorScaling: bilinear
videoMonitorUse444SDI: 0
videoMonitorUseHDROverHDMI: 0
videoMonitorUseLevelA: 0
videoMonitorUseMatrixOverrideFor422SDI: 0
videoMonitorUseStereoSDI: 0
videoPlayoutAudioFramesOffset: 0
videoPlayoutBatchHeadDuration: 8
videoPlayoutBatchTailDuration: 8
videoPlayoutLTCFramesOffset: 0
videoPlayoutMode: video_audio
videoPlayoutShowLTC: 0
videoPlayoutShowSourceTimecode: 0
superScale: 1
"""

# Process each line to create define values list

# Splitting the input text into lines and filtering out empty lines
lines = text_dump.strip().split("\n")


# Defining a function to convert camel case to uppercase snake case
def camel_to_upper_snake(camel_str):
    # Inserting an underscore before capital letters and making everything uppercase
    snake_str = ''.join(['_' + i.lower() if i.isupper() else i for i in camel_str]).lstrip('_').upper()
    return snake_str


# Processing each line according to the given steps
for line in lines:
    key, _ = line.split(": ", 1)  # Splitting each line into key and value
    define_key = camel_to_upper_snake(key)
    define_statement = f'{define_key} = "{key}"'  # Creating the define statement
    print(define_statement)
