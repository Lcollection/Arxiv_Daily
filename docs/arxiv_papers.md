# Arxiv Papers

| Title | Summary | PDF Link | Code Link | Translated Title | Translated Summary | Summary |
|-------|---------|----------|-----------|------------------|--------------------|---------|
| Bridging Episodes and Semantics: A Novel Framework for Long-Form Video Understanding | While existing research often treats long-form videos as extended short
videos, we propose a novel approach that more accurately reflects human
cognition. This paper introduces BREASE: BRidging Episodes And SEmantics for
Long-Form Video Understanding, a model that simulates episodic memory
accumulation to capture action sequences and reinforces them with semantic
knowledge dispersed throughout the video. Our work makes two key contributions:
First, we develop an Episodic COmpressor (ECO) that efficiently aggregates
crucial representations from micro to semi-macro levels. Second, we propose a
Semantics reTRiever (SeTR) that enhances these aggregated representations with
semantic information by focusing on the broader context, dramatically reducing
feature dimensionality while preserving relevant macro-level information.
Extensive experiments demonstrate that BREASE achieves state-of-the-art
performance across multiple long video understanding benchmarks in both
zero-shot and fully-supervised settings. The project page and code are at:
https://joslefaure.github.io/assets/html/hermes.html. | [PDF](http://arxiv.org/pdf/2408.17443v1) | N/A | 跨越剧集与语义：一种理解长篇视频的新框架 | 尽管现有研究常将长视频视为扩展的短视频，我们提出了一种更准确反映人类认知的新方法。本文介绍了BREASE：一种用于长视频理解的模型，通过模拟情节记忆积累来捕捉动作序列，并通过视频中分散的语义知识对其进行强化。我们的工作有两个关键贡献：首先，我们开发了情节压缩器（ECO），能有效地从微观到半宏观层面聚合关键表征；其次，我们提出了语义检索器（SeTR），通过关注更广泛的上下文，增强这些聚合表征的语义信息，显著降低特征维度同时保留相关的宏观级别信息。大量实验表明，BREASE在零样本和完全监督设置下，在多个长视频理解基准测试中达到了最先进的性能。项目页面和代码位于：https://joslefaure.github.io/assets/html/hermes.html。 | 关键词：长视频理解，BREASE模型，情节记忆积累，情节压缩器（ECO），语义检索器（SeTR），零样本学习，完全监督学习，最先进性能。 |
| SelectTTS: Synthesizing Anyone's Voice via Discrete Unit-Based Frame Selection | Synthesizing the voices of unseen speakers is a persisting challenge in
multi-speaker text-to-speech (TTS). Most multi-speaker TTS models rely on
modeling speaker characteristics through speaker conditioning during training.
Modeling unseen speaker attributes through this approach has necessitated an
increase in model complexity, which makes it challenging to reproduce results
and improve upon them. We design a simple alternative to this. We propose
SelectTTS, a novel method to select the appropriate frames from the target
speaker and decode using frame-level self-supervised learning (SSL) features.
We show that this approach can effectively capture speaker characteristics for
unseen speakers, and achieves comparable results to other multi-speaker TTS
frameworks in both objective and subjective metrics. With SelectTTS, we show
that frame selection from the target speaker's speech is a direct way to
achieve generalization in unseen speakers with low model complexity. We achieve
better speaker similarity performance than SOTA baselines XTTS-v2 and VALL-E
with over an 8x reduction in model parameters and a 270x reduction in training
data | [PDF](http://arxiv.org/pdf/2408.17432v1) | N/A | SelectTTS：基于离散单元帧选择合成任何人声音 | 合成未见说话者的声音是多说话者文本到语音转换（TTS）中持续存在的挑战。大多数多说话者TTS模型依赖于在训练期间通过说话者条件建模说话者特征。通过这种方法对未见说话者属性进行建模已导致模型复杂性的增加，这使得再现结果和改进结果变得具有挑战性。我们设计了一个简单的替代方案。我们提出了SelectTTS，一种新颖的方法，从目标说话者中选择适当的帧，并使用帧级自监督学习（SSL）特征进行解码。我们展示了这种方法可以有效地捕捉未见说话者的说话者特征，并在客观和主观指标上与其他多说话者TTS框架取得可比的结果。通过SelectTTS，我们展示了从目标说话者的语音中选择帧是一种直接的方式，可以在模型复杂度较低的情况下实现对未见说话者的泛化。我们实现了比SOTA基线XTTS-v2和VALL-E更好的说话者相似性性能，模型参数减少了8倍以上，训练数据减少了270倍。 | 多说话者TTS、未见说话者、模型复杂性、SelectTTS、帧级SSL特征、泛化能力、说话者相似性、参数减少、训练数据减少。 |
