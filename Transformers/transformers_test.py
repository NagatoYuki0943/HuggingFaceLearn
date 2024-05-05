# PreTrained

from transformers import (
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    PretrainedConfig,
    PreTrainedModel,
)


# pipeline

from transformers import pipeline


# Auto

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoModelForVisualQuestionAnswering,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
)


# generation config

from transformers import (
    GenerationConfig,
)


# AUDIO MODELS

## MMS
from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2ForSequenceClassification,
)

## Whisper
from transformers import (
    WhisperProcessor,
    WhisperConfig,
    WhisperModel,
    WhisperFeatureExtractor,
    WhisperForAudioClassification,
    WhisperForConditionalGeneration,
)

## SeamlessM4T
from transformers import (
    SeamlessM4TTokenizer,
    SeamlessM4TTokenizerFast,
    SeamlessM4TProcessor,
    SeamlessM4TConfig,
    SeamlessM4TModel,
    SeamlessM4TForTextToText,
    SeamlessM4TForTextToSpeech,
    SeamlessM4TForSpeechToSpeech,
    SeamlessM4TForSpeechToText,
    SeamlessM4TPreTrainedModel,
    SeamlessM4THifiGan,
    SeamlessM4TCodeHifiGan,
)


# MULTIMODAL MODELS

## Blip2
from transformers import (
    Blip2Processor,
    Blip2Config,
    Blip2Model,
    Blip2ForConditionalGeneration,
    Blip2VisionConfig,
    Blip2VisionModel,
    Blip2QFormerConfig,
    Blip2QFormerModel,
    OPTConfig,
)

## Blip
from transformers import (
    BlipProcessor,
    BlipImageProcessor,
    BlipConfig,
    BlipModel,
    BlipTextConfig,
    BlipTextModel,
    BlipVisionConfig,
    BlipVisionModel,
    BlipForConditionalGeneration,
    BlipForImageTextRetrieval,
    BlipForQuestionAnswering,
)

## ChineseCLIP
from transformers import (
    ChineseCLIPProcessor,
    ChineseCLIPConfig,
    ChineseCLIPModel,
    ChineseCLIPTextConfig,
    ChineseCLIPTextModel,
    ChineseCLIPVisionConfig,
    ChineseCLIPVisionModel,
)

## CLIP
from transformers import (
    CLIPTokenizer,
    CLIPTokenizerFast,
    CLIPConfig,
    CLIPModel,
    CLIPTextConfig,
    CLIPTextModel,
    CLIPVisionConfig,
    CLIPVisionModel,
    CLIPProcessor,
    CLIPTextModelWithProjection,
    CLIPVisionModelWithProjection,
)

## Data2Vec
from transformers import (
    Data2VecAudioConfig,
    Data2VecAudioModel,
    Data2VecAudioForAudioFrameClassification,
    Data2VecAudioForCTC,
    Data2VecAudioForSequenceClassification,
    Data2VecAudioForXVector,
    Data2VecTextConfig,
    Data2VecTextModel,
    Data2VecTextForCausalLM,
    Data2VecTextForMaskedLM,
    Data2VecTextForMultipleChoice,
    Data2VecTextForQuestionAnswering,
    Data2VecTextForSequenceClassification,
    Data2VecTextForTokenClassification,
    Data2VecVisionConfig,
    Data2VecVisionModel,
    Data2VecVisionForImageClassification,
    Data2VecVisionForSemanticSegmentation,
)

## Perceiver
from transformers import (
    PerceiverTokenizer,
    PerceiverConfig,
    PerceiverModel,
)
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverTextPreprocessor,
    PerceiverImagePreprocessor,
    PerceiverAudioPreprocessor,
)

## Vilt
from transformers import (
    ViltProcessor,
    ViltImageProcessor,
    ViltConfig,
    ViltModel,
    ViltForImageAndTextRetrieval,
    ViltForImagesAndTextClassification,
    ViltForMaskedLM,
    ViltForQuestionAnswering,
    ViltForTokenClassification,
)

## Owlv2
from transformers import (
    Owlv2ImageProcessor,
    Owlv2Processor,
    Owlv2TextConfig,
    Owlv2TextModel,
    Owlv2VisionConfig,
    Owlv2VisionModel,
    Owlv2Config,
    Owlv2Model,
    Owlv2ForObjectDetection,
)

## Kosmos2
from transformers import (
    Kosmos2Config,
    Kosmos2Processor,
    Kosmos2Model,
    Kosmos2ForConditionalGeneration,
)


# TEXT MODELS

## Bart
from transformers import (
    BartTokenizer,
    BartTokenizerFast,
    BartConfig,
    BartModel,
    BartForCausalLM,
    BartForConditionalGeneration,
    BartForQuestionAnswering,
    BartForSequenceClassification,
)

## Bert
from transformers import (
    BertTokenizer,
    BertTokenizerFast,
    BertConfig,
    BertModel,
    BertForMaskedLM,
)

## Bert
from transformers import (
    GPT2Tokenizer,
    GPT2TokenizerFast,
    GPT2Config,
    GPT2Model,
    GPT2DoubleHeadsModel,
    GPT2ForQuestionAnswering,
    GPT2ForSequenceClassification,
    GPT2ForTokenClassification,
    GPT2LMHeadModel,
)

## T5
from transformers import (
    T5Tokenizer,
    T5TokenizerFast,
    T5Config,
    T5Model,
    T5EncoderModel,
    T5ForConditionalGeneration,
    T5ForQuestionAnswering,
)

# Reformer
from transformers import (
    ReformerTokenizer,
    ReformerTokenizerFast,
    ReformerConfig,
    ReformerModel,
    ReformerModelWithLMHead,
    ReformerForMaskedLM,
    ReformerForSequenceClassification,
    ReformerForQuestionAnswering,
)

# RWKV
from transformers import (
    RwkvConfig,
    RwkvModel,
    RwkvForCausalLM,
)


# TIME SERIES MODELS

## Autoformer
from transformers import (
    AutoformerConfig,
    AutoformerModel,
    AutoformerForPrediction,
)
