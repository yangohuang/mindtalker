#!/usr/bin/env bash
# Batch generate paired TTS features for BridgeMLP training data prep.
# Strategy: 50 Chinese sentences × 9 voices = 450 TTS clips → wav2vec2 features.
# Goal: build P(synthetic) distribution for BridgeMLP's unpaired domain adaptation.
#
# This produces: data/tts_batch/{voice}/clip_{idx:03d}.{wav,npy}
#
# Companion: real audio side will use existing podcast_sichuan + (later) AISHELL-3.
# For now P(real) is single-clip; batch will follow.

set -e
cd /home/yg/yg/code/mindtalker/research

OUT_DIR=data/tts_batch
mkdir -p "$OUT_DIR"

# 50 prompts: mix of conversational, news-like, descriptive Chinese
# Manually curated for variety in syntax / content / length (3-10s each)
SENTENCES=(
  "今天天气不错，适合出门散步，我们去公园走一走吧。"
  "最近的人工智能技术发展非常迅速，特别是大语言模型领域。"
  "这家咖啡店的拿铁口感很好，价格也不贵，推荐给大家。"
  "我们公司昨天开了一个会议，主要讨论了下一季度的产品规划。"
  "深圳的春天总是来得比较早，三月初就能感受到温暖的气息。"
  "周末打算去爬山，听说那边的风景特别美，还有瀑布可以看。"
  "他写的代码风格非常清晰，注释也很详细，是个值得学习的同事。"
  "这本小说讲了一个关于时间旅行的故事，情节非常吸引人。"
  "我觉得这种新型的电动车设计很巧妙，续航能力也比上一代强。"
  "明天下午有个重要的客户拜访，需要提前准备好所有的演示材料。"
  "这次的实验结果出乎意料，模型在小数据集上表现得相当不错。"
  "妈妈做的红烧肉是家里最好吃的菜，每次回家都要点这一道。"
  "篮球比赛打到加时赛才分出胜负，最后一秒的三分球真是太精彩了。"
  "这部电影的特效做得很逼真，票房应该能突破十亿元。"
  "学习一门新的编程语言需要时间，但是收获也是巨大的。"
  "国庆假期我打算回老家看望父母，已经买好了高铁票。"
  "最近读了一本关于宇宙学的书，里面介绍了暗物质的研究进展。"
  "这家餐厅的服务态度特别好，菜品的摆盘也很有艺术感。"
  "公司今年准备扩大研发团队，预计要招聘三十名工程师。"
  "夜晚的城市灯光闪烁，远远望去就像一片星海。"
  "孩子们在草地上追逐打闹，笑声传到了整个公园。"
  "这首钢琴曲的旋律非常优美，让人听了心情平静。"
  "新发布的智能手机配置很高，但是价格也比之前贵了不少。"
  "他的演讲很有感染力，听众都被深深打动了。"
  "我们决定在元旦期间举办一个产品发布会，邀请媒体和合作伙伴。"
  "这道数学题看起来简单，实际上需要多个步骤才能解出来。"
  "外婆家的院子里种着几棵桂花树，每到秋天就香气四溢。"
  "这次出差去了三个城市，认识了很多有意思的人。"
  "图书馆里非常安静，是个适合学习和思考的地方。"
  "新装修的办公室明亮宽敞，员工的工作效率明显提升了。"
  "这场雨下了整整一夜，第二天早上路面上到处都是积水。"
  "他的研究方向是量子计算，最近发表了一篇高水平的论文。"
  "周末和朋友一起做饭，亲手烹饪的食物总是格外美味。"
  "海边的日落特别壮观，天空被染成了橙红色。"
  "这次旅行让我看到了不同的文化，开阔了眼界。"
  "运动是保持健康的最好方式，每周至少要锻炼三次。"
  "这家书店的选书很有品味，是我每周必去的地方。"
  "新员工培训持续了两周，主要内容包括公司文化和业务流程。"
  "山上的空气格外清新，深呼吸一口都觉得心旷神怡。"
  "这次疫情让大家更加重视健康，也促进了远程办公的发展。"
  "古典音乐会的氛围非常庄严，演奏家的水平令人钦佩。"
  "工程师文化推崇务实和创新，这两点缺一不可。"
  "孩子的成长是父母最关心的事情，每个阶段都需要不同的引导。"
  "这款游戏的画面精美，剧情也很有深度，值得花时间去玩。"
  "深夜的便利店总是开着，给加班的人带来一丝温暖。"
  "他在十年前创立了这家公司，从一个想法发展到了现在的规模。"
  "园艺是一种很好的爱好，既能锻炼身体又能陶冶情操。"
  "新闻里说今年的经济增长超过了预期，各行各业都有新的机会。"
  "在这个快节奏的时代，留出一些时间给自己思考是非常重要的。"
  "回顾过去一年的工作，有进步也有需要改进的地方。"
)

VOICES=(vivian serena dylan eric ryan aiden uncle_fu sohee ono_anna)

# Wave 1: only first 25 sentences × 5 voices = 125 clips, ~10 minutes total
# Tunable; can extend later.
N_SENT=${N_SENT:-25}
N_VOICE=${N_VOICE:-5}

echo "[batch] Generating ${N_SENT} sentences × ${N_VOICE} voices = $((N_SENT * N_VOICE)) clips"
echo "[batch] Output: $OUT_DIR/"

idx=0
for v in "${VOICES[@]:0:$N_VOICE}"; do
  mkdir -p "$OUT_DIR/$v"
  for ((i=0; i<N_SENT; i++)); do
    text="${SENTENCES[$i]}"
    out_wav="$OUT_DIR/$v/clip_$(printf '%03d' $idx).wav"
    if [ -f "$out_wav" ]; then
      echo "  [skip] $out_wav exists"
    else
      response=$(curl -s -X POST http://127.0.0.1:8200/v1/audio/speech/file \
        -H "Content-Type: application/json" \
        -d "{\"input\": \"$text\", \"voice\": \"$v\", \"response_format\": \"wav\"}")
      tmp_path=$(echo "$response" | python3 -c "import json,sys; print(json.load(sys.stdin)['file_path'])")
      if [ -n "$tmp_path" ] && [ -f "$tmp_path" ]; then
        cp "$tmp_path" "$out_wav"
        echo "  [ok] [$v][$idx] $(basename $out_wav) $(stat -c '%s' $out_wav) bytes"
      else
        echo "  [FAIL] $v $idx: $response"
      fi
    fi
    idx=$((idx + 1))
  done
done

echo ""
echo "[batch] Done. $(find $OUT_DIR -name '*.wav' | wc -l) wav files generated."
