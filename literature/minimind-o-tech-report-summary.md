---
title: "MiniMind-O开源 | 0.1B参数，1个人+4块3090用4小时，造了个全模态模型"
author: 多模态实验室
date: "2026-05-08 08:31:03"
source: "https://mp.weixin.qq.com/s/OqbLTvM2eX62fguVU5_1qw"
---

# MiniMind-O开源 | 0.1B参数，1个人+4块3090用4小时，造了个全模态模型

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/hzCR32CiabkF8M6WjSMWiaBibMDvPZjLv6vEWxh7bj4XqS3tes8cEwvibb1Gs9aicOI5j4elficWPGbQV92elH9MewRWXOMWib5Fxhy1uvpKYYOj04/640?from=appmsg#imgIndex=0)

扫描下方二维码，添加交流群深入交流

![图片](https://mmbiz.qpic.cn/mmbiz_png/hzCR32CiabkHNvQFxt9btndATabSAQHELYFs4icnb4QfclAuzHcjic2k2zY62zrgriax74bcLJXdcgEXdFL1ZGMIp9hwJnTbTOdyBBmUYpQicT7g/640?wx_fmt=png&from=appmsg&wxfrom=5&wx_lazy=1&tp=webp#imgIndex=1)

你还在用ASR+LLM+TTS的级联方案做语音交互吗？延迟高、错误难追溯、说话人控制割裂——一个0.1B参数的开源全模态模型MiniMind-O直接砍掉所有中间模块，实现文本、语音、图像输入到流式语音输出的完整闭环，在消费级GPU上四小时就能复现，CER最低仅0.0897！

🔥 **开源代码已放出**：https://github.com/jingyaogong/minimind-o[1]

但为什么99%的语音交互系统还在用级联？因为全模态设计在小模型上太难了——每个新增模态都要挤过极窄的隐藏空间。MiniMind-O用0.1B参数做了一次“压力测试”，结果发现：中间层桥接、低秩码本接口、分阶段训练数据，这三个设计选择直接决定了小模型能否跑通完整语音循环。

**💡 核心设计：Thinker-Talker架构**

传统级联方案把语音理解（ASR）、语义推理（LLM）和语音生成（TTS）拆成三个独立模块，每个模块的错误会叠加放大。MiniMind-O反其道而行：用一个完整的MiniMind Transformer作为Thinker负责多模态理解，再加一个独立的四层Talker负责流式语音生成。Thinker和Talker共享同一个隐藏空间，通过Hidden Bridge连接。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/hzCR32CiabkFJksDCGs1EYUA8ibVicnoutvvdacffJLFAyxQ9lAq5tXiaOrctAZKjD1liaO0txiaeicrU67GBH6wahjsUDQOTWcTxXxJtUHr4cbYFk/640?from=appmsg#imgIndex=2)图：MiniMind-O整体架构——输入端的音频（SenseVoice）、图像（SigLIP2）、文本（Tokenizer）通过MLP投影器映射到统一隐藏空间，Thinker理解后通过Hidden Bridge将中间层状态传给Talker，Talker自回归生成8层Mimi码本，最终解码为24kHz语音\*

这个设计的精妙之处在于：Talker不是LLM的简单后缀，它同时读取Thinker的中间层状态和自回归的音频码历史。这就好比一个同声传译员，一边听演讲者的语义（Thinker状态），一边看自己刚才写的笔记（音频码历史），两者融合后说出流利的译文。

**💡 中间层桥接：为什么不是最后一层？**

你可能觉得，Talker应该读取Thinker的最后一层输出，因为那包含了最丰富的语义信息。但实验发现：最后一层已经过度适配了下一个文本token的分类器，它的隐藏状态携带了LM头的几何偏差，对声学条件来说是噪声。而嵌入层又太浅，还没积累足够的上下文来处理发音、句法和跨模态指代。

举个例子：中文汉字“地”，在“土地”里读dì，在“慢慢地”里读de。嵌入层只知道这是个字，不知道上下文；最后一层已经决定了下一个词是什么，但把声学信息挤掉了。中间层（默认第3层，总8层）恰好平衡——既有足够的上下文，又没有完全坍缩到文本分类器。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/hzCR32CiabkFTmVqEZDEK3kQzA8IkyYN4o0PnfExHR814A7OY6yaCEAmQpiaFDoicib0r9t8nMtP21bibNqfl1NcUZXdEVGs9w66zia1ynyPYgvas/640?from=appmsg#imgIndex=3) \*表：Talker隐藏层维度消融实验——768维是最优平衡点，降到384维时MoE变体CER从0.0900飙升到0.1285，参数节省远不足以弥补一致性损失\*

**💡 低秩码本接口：参数效率的关键**

八个Mimi码本如果每个都配独立的嵌入表和输出头，参数量会爆炸。MiniMind-O采用共享基座+低秩Adapter的方案：嵌入层用共享的嵌入表加上每个码本的低秩Adapter，输出头用共享线性头加上每个码本的低秩Adapter。秩消融实验表明，中等秩（r=64）就能恢复大部分性能，而且输出头的秩比嵌入层的秩更重要。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/hzCR32CiabkEgOzJSIEG5o18VAVCqP8xrWDU48UiaX5vjCjqqu71CoiaZxHz3QHtIhiaFaQBXoJUtrET5N0HWYbgay1YYFZMQ7YsRfg4UMCAwOA/640?from=appmsg#imgIndex=4) \*图：秩消融实验——统一秩r=64时音频损失已接近满秩水平，解耦实验显示输出头秩（Head rank）比嵌入层秩（Embed rank）对性能影响更大\*

**💡 训练流程：三阶段渐进式对齐**

从单模态到多模态，MiniMind-O采用三阶段训练策略：Stage1文本到语音（T2A）微调，仅更新Thinker和Talker；Stage2引入视觉和音频编码器，通过投影器对齐跨模态特征；Stage3统一多模态指令微调，整合文本、音频、图像三模态输入。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/hzCR32CiabkHCrVMdCj7bnFEvb1ibicuEF5hUibNpv4M7aUTZoe1jTicIZBbD0IPZic8jbQhcZ3fkfzuzhd5QAB4wAsAJj4WOicadL8YAzfmOyCCI4/640?from=appmsg#imgIndex=5) \*图：三阶段训练流程——Stage1 T2A语音生成微调，Stage2跨模态感知对齐，Stage3多模态指令微调，投影器在跨模态对齐中起关键作用\*

全部训练在四块RTX 3090上四小时内完成，这得益于0.1B的活跃参数规模。你可能会问：这么小的模型能做什么？看看下面的实时交互流程。

**💡 流式解码与打断机制**

MiniMind-O支持真正的流式语音交互：第一个文本token生成后即可开始播放音频，无需等待完整回答。更酷的是基于VAD的打断机制——用户新发言时模型自动中止当前输出并启动新回复。

![Image](https://mmbiz.qpic.cn/mmbiz_jpg/hzCR32CiabkEicasKOUleLZib51KBz3ViajJqO9HQic82icUn1bPdTNXJ5jPJw8oYf0u9YqU0oD1HEyCNY1ADJ60bxpZszqTibEvqNzcnN7HVNaWRo/640?from=appmsg#imgIndex=6) \*图：实时交互时序——从用户语音输入到模型回复，TTFT仅140ms，TTFA仅260ms，VAD检测到用户打断后立即中止并启动新回复\*

**📊 实验验证：数据说话**

在跨模型英文文本到音频（T2A）一致性评估中，MiniMind-O的minimind-3o变体在参数量仅0.1B的情况下，平均CER 0.0897，对比Mini-Omni（0.5B）的0.010和Mini-Omni2（0.5B）的0.012，差距并不大。更难得的是，在20个具体问题的逐题对比中，14个问题实现了零CER。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/hzCR32CiabkENZVAHiaDlnGBnFcUpWic8CJs5Ck23B9lHWDJGkDDQMoQyYnROvhd25BgRG71TElETdia9uiczvckgWiaN84UHJu6r2ZDfsnnqia4do/640?from=appmsg#imgIndex=7) \*表：跨模型T2A一致性对比——minimind-3o在0.1B参数下达到0.0897 CER，与0.5B的Mini-Omni系列差距可接受\*

语音克隆方面，minimind-3o在未见说话人上的CAM++余弦相似度达到0.5995，优于先前基线。视觉-语言到语音任务中，模型能准确描述图像内容并生成对应语音。

![Image](https://mmbiz.qpic.cn/sz_mmbiz_jpg/hzCR32CiabkHEhwkicl5MlwK0v2iaB95lKC5szHGXxDpzGjOje9icBDkgyTq3NicMJ9Vyx4XJW8zKXJ1ZrMfp3RnyiaPD3icdCuy3nzAXIAud66lics/640?from=appmsg#imgIndex=8) \*图：A2A定性示例——语音输入“为什么天空是蓝色的？”模型生成文本并语音输出，中英文混合场景下依然流畅\*  ![Image](https://mmbiz.qpic.cn/mmbiz_jpg/hzCR32CiabkGfqjN9tzujTWL20udt4dVibFOI4GEeR1tLHUyBMUGTEpSVwso7jLJebKQtB6iaiahAHjXxG7qVYIPlgfZKVjDZicfsx82IC7kiab3M/640?from=appmsg#imgIndex=9) \*图：图像到音频生成——输入猫、宇航员骑车等图像，模型生成对应文本描述并语音输出\*

**⚖️ 客观评价：小模型的局限与价值**

语音自然度和长句稳定性仍落后于更大模型，中等长度英文回答是最薄弱环节。视觉通路使用固定SigLIP2编码器和简单MLP投影器，远非大型VLM的替代。语音克隆高度依赖参考质量。但MiniMind-O的核心价值不是与前沿模型竞争，而是提供一个完全可复现、可检查的小规模基线，让全模态设计的关键选择不再隐藏在规模背后。

🤔 **深度思考**：你认为这项技术最可能颠覆哪个AI应用场景？比如实时语音助手、教育互动、还是无障碍通信？**欢迎在评论区留下你的观点！**

💝 **支持原创**：如果本文帮到你，**点赞+在看**就是最好的支持！**分享**给你的技术伙伴！

🔔 **关注提醒**：设为星标，第一时间获取深度技术解读！

#AI技术 #深度学习 #模型优化 #技术干货 #论文解读

## 参考

MiniMind-O Technical Report: An Open Small-Scale Speech-Native Omni Model

#### 引用链接

`[1]`: *https://github.com/jingyaogong/minimind-o*
