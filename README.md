# Share Analysis

Share是自己做股票的小工具（目前只支持深沪两市）。Share Analysis试图使用机器学习方法挖掘股票价值。

## 随想

在尝试进入股票市场后，我心中一直有个很大的疑问：股票价格究竟与什么有关？

有人说公司的资产价值（不仅仅是总资产，还包括偿债能力、盈利能力、成长能力等等）影响股价。但财务报表一季度发一次，在此期间也不是按照一个方向运动，很明显不能作为交易指标。如果说都是幕后消息造成的，那么所有散户都是韭菜，我认为这并不合理。所以得出“资产价值只要不超过某一范围就应该被认为合理”的结论。

有人说公司分红影响股价。我觉得这个比较靠谱，为此还专门写了一篇博客——[从公司分红看股票价值](https://frederchen.com/article/ff80818172ade55d0172adf97fdc0003)，最后得出有一点点影响的结论。从博文中可以很明显的看出，股票市场的利润大头还是卖出和买入差价。尤其在国内市场，分红由公司自己来决定，过去分红多不代表未来分红多，甚至未来都不一定有分红。所以只能说过去有分红的公司风险较小，但作为交易指标有点牵强。

有人说市场（也就是大环境，比如国内经济因素）能带动所有股票齐涨齐跌。也就是在说：市场齐涨叫牛市，市场齐跌叫熊市，市场涨跌互现叫震荡市。市场所处周期很难判断，齐跌完出现反弹应该是认为牛市还是震荡市完全靠个人经验。如果无法分辨市场当时所处周期，那么所处周期只能是后验的。

有人说公司所在地域、行业影响股价。这个观念和上一条的市场论相似，甚至应该认为就是细分以后的市场论。使用这个方法需要有一些眼界，我的经验是从生活中发现价值，比如AirPods刚出来的时候搜索发现“立讯精密”是生产商果断买入、疫情期间医药行业公司会鸡犬升天。这是目前实践下来比较有用的观点。另外，该观点有一个很大的缺陷在于市场存在错配情况，导致市场有时不会立马验证判断的正确性，这样就无法判断究竟是市场错了还是自己错了。也由于这个缺陷，很多人从做交易转向哲学……

还有人说市场大众的心理因素影响股价。这是在说：市场买方多于卖方就涨，反之就跌。我认为这是最接近真实股价的逻辑。问题在于如何才能知道大众对市场的看法，为此引申出了技术分析——企图从历史交易数据中发现大众对市场的看法。

基于以上观念，目前自己的交易大致流程：公司资产不存在大问题、历史上有稳定分红就可入选观察范围；个人判断公司可能增值就开始关注股价走势；大众开始对市场看多（也就是技术分析层面看多）才决定入场。

说回Share Analysis。当前的Share Analysis是不完善的项目。只使用各平台分析数据来做排序，人为盯盘还是必不可少的步骤。使用各平台的分析数据是因为我发现家里人买股票非常迷信这种“高级”指标，而这一指标并不在我的系统范围内，为了简化流程，稍做了一些编码工作来实现局部自动化。未来可能会加入更多的新功能。

## 原理

各平台数据一般是在交易日晚上更新，并且无法访问历史数据，所以写爬虫并将数据保存在本地是必须的流程。爬虫项目目前不准备开源（能用但是没写好），我写了简单的后台接口（在项目中设为了默认数据请求源）希望不会被爆破。

各平台的数据基本都是分数，可直接作为特征使用。Label做了一些尝试，不知道能不能给出更好的额答案（有想法欢迎讨论）。由于国内交易需要T+1，选择了“第一天的数据使用第三天的收盘价减去第二天的开盘价大于0标1否则标0”作为Label。

在训练的时候使用前三十天的数据制作模型，预测后一天的情况。测试时需要注意，倒数第二天不能加入训练集，因为预测时也是没有这一天的，如果加入可能造成测试、预测结果不一致的情况，甚至某种程度上算数据穿越。

在做这个模型之前我一直考虑股票价格究竟与什么有关，

## 安装

1. 安装Python3

2. 安装[LightGBM模型包](https://lightgbm.readthedocs.io/en/latest/Installation-Guide.html)

3. 安装Python库

   ```shell
   python3 -m pip install requests numpy pandas sklearn lightgbm matplotlib fire
   ```

## 功能及使用

```shell
# 打开cmd输入, n为训练数据日期跨度（默认30），end为最后一天日期（默认today）
python3 -m share_analysis test --n 30 --end yyyymmdd # 测试: end 为pred lebel的日期
python3 -m share_analysis pred --n 30 --end yyyymmdd # 预测: end 为pred feature的日期
```

同时会对数据做一些缓存，目录在utils.py中，对应代码：
```python
TMP_PATH = os.path.join(tempfile.gettempdir(), 'share_analysis')
```

## 有趣的链接

- [项目链接](https://github.com/Freder-chen/share_analysis)
