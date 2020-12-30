---
layout: math
title: GraphHomomorphism Convolution解説
tags: 
- machine learning
- GNN
---

[Graph Homomorphism Convolution](https://arxiv.org/pdf/2005.01214.pdf)を調べたメモです.
どうしても調べたりない部分があると思うので、誤りはコメントいただけるとありがたいです。
論文が全体的に読みやすいので気になる人は原論文を読んでください.


## 論文のSummary
- Graph Homomorphismの個数を使った解析手法を導入
- Weighted Graphの場合も含め、いくつかの場合に普遍近似定理を証明
- 証明のポイントはStone Weierstrassの定理とその条件を満たす空間を作ること.
- 実応用時にNNを使うこともできるが、別に使わなくてもよい.
- 実験でGINで学習できないデータでも高い精度を出すことができた.

論文ではContirbutionは以下とされていました.

> - Introduce and analyze the usage of weighted graph homomorphism numbers with a general choice of F. The choice of F is a novel way to parameterize the capability of graph learning models compared to choosing the tensorization order in other related work.
> - Prove the universality of the homomorphism vector in approximating F-indistinguishable functions. Our main proof technique is to check the condition of the Stone-Weierstrass theorem.
> - Empirically demonstrate our theoretical findings with synthetic and benchmark datasets. Notably, we show that our methods perform well in graph isomorphism test compared to other machine learning models.


以下の問題が基本的な考察対象になります.

|Porblem(グラフ分類問題)|
|:--|
|$$ \{G_i, x_i, y_i\}_{i=1, \ldots, n}, x_i: V(G_i) \to \mathcal{X}, y_i \in \mathcal{Y}$$に対し$$h(G_i, x) = y$$となる仮説$$h$$を学習できるか.|||

最初に記号の説明をすると,

- $$G_i$$はグラフ
- $$x_i$$はグラフのVetex上の特徴量を指定するものです.各Vertexに対し,特徴量が対応するので$$G_i$$のVertexの集合$$V(G)$$から$$\mathcal{X}$$への写像とみなしています.
- $$\mathcal{Y}$$は正解ラベルの空間で集合としては$$\mathcal{Y} =  \{0, 1 \}$$を考えます.


とはいえ,実際にはこの論文では上の問題を明に扱うわけではありません.
基本的にはGraph Embedding.
つまり,$$G \to \mathbb{R}^n$$を作ることを目指します.$$\mathbb{R}^n \to  \{0, 1\}$$は機械学習で典型的に扱われる対象なので,適切に(効率的に)Graphが埋めれ込めればよいという考え方です.

当然,何が適切な埋め込み方法が必要なわけですが,
グラフの分類問題などではinductive biasとしてvertexの置き換えに対して不変であることを仮定するので,
ここではそれを一般的な形でまとめ,$$\mathcal{F}$$-invariantという形で考えます.($$\mathcal{F}$$-invariantの定義は後述.)

ここでこの後も使われるグラフ関係の用語を定義しておきます.論文とは順序が異なりますが,Weightedの場合も定義しておきます.

## 用語の定義

### グラフの基本的な用語

- グラフという場合ここではundirected グラフを考えます.
- グラフ$$G$$に対し,Vetexを$$V(G)$$,Edgeを$$E(G)$$で表します.
- グラフが __単純__ とは多重辺やloopの存在しないグラフのことです.
- $$\sigma: V(G) \to U$$が全単射のとき,$$G^{\sigma}$$を$$V(G^{\sigma})= U, E(G^{\sigma}) =  \{(\sigma(u), \sigma(v) \mid (u, v) \in E(G) \}$$とします.
- $$G_1$$と$$G_2$$が同型とはある全単射$$\sigma: V(G_1) \to V(G_2)$$が存在し,$$G_1^{\sigma} = G_2$$となること.
- $$\mathcal{G}$$で単純グラフ全体を表す.

次に今回出てくるHomomorphism関係の用語を定義します.
### Homomorphism numbers
- $$\mathrm{Hom}(F,G) =  \{f \in map(V(F),V(G)) \mid  (u,v) \in E(F)ならば (f(u), f(v))\in E(G) \} $$
- $$\mathrm{hom}(F,G) = \vert \mathrm{Hom}(F,G) \vert$$
- $$t(F, G) = \mathrm{hom}(F, G)/ |V(G)|^{|V(F)|} =  \sum_{\pi:V(F) \to V(G)} \displaystyle\prod_{u \in V(F) }\frac{1}{|V(G)|} \prod_{(u, v) \in E(F)} 1_{[(\pi(u), \pi(v)) \in E(G)]}$$
  - $$\sum$$はただのmap
  - $$\prod_{u \in V(F)}$$は$$F$$側の全てのノードの積
  - 次は$$u$$と連結している全てのノードの積,2つの積で全て1になっていることがグラフのhomomorphismになっていることと一致している.


グラフの同型性は上で定義したHomomorphismを使って判定できます.

|Theorem|
|:--|
|$$G_1 \simeq G_2$$は全ての単純グラフ$$F$$に対し,$$hom(F, G_1) = hom(F, G_2)$$と同値である.|||

というわけでグラフの分類をHomomorphismを使ってやりたくなるモチベーションがあります。

__remark__
$$G_1,G_2$$は制限を書いていませんが、単純グラフなのだと思います。

## Homomorphism NumbersとGraphの近似

全てのグラフを調べるのは実用的ではないので,
都合のいいグラフの集合$$\mathcal{F} \subset \mathcal{G}$$を選びそれを使って学習させることを考えます。

|Definition|
|:--|
|$$\forall F \in \mathcal{F}$$に対し, $$hom(F,G_1) = hom(F,G_2)$$となる時, $$\mathcal{F}$$- __indistinguable__ という.|||

|Definition|
|:--|
|$$f: \mathcal{G} \to \mathbb{R}$$が$$\mathcal{F}$$-__invariant__ とは$$G_1,G_2 \in \mathcal{G}$$が$$\mathcal{F}$$-indistinguableの時に$$f(G_1) = f(G_2)$$となることをいう.|||

<!-- https://web.cs.elte.hu/~lovasz/bookxx/hombook-almost.final.pdf -->

今回の論文ではグラフに対して普遍近似定理を示すものです.
ベースとなる方針について説明します.

普遍近似定理はいくつか証明方法があるかと思いますが、その一つにStone-Weierstrassの定理と呼ばれる定理を使って示す方法があります。

これはNN全体の集合$$A$$が特定の条件を満たせば、連続関数をいくらでも近似できることを保証するものです。

なので、この論文ではもとの空間と近似する側の$$A$$を定めた後実際に$$A$$について条件を確認し、普遍性を示します。


Stone-Weierstrassの定理の主張とその主張で出てくる用語を説明します。


ある(位相)空間$$X$$から$$\mathbb{R}$$への連続写像全体を$$C(X, \mathbb{R})$$とします.その部分集合$$A \subset C(X, \mathbb{R})$$が$$X$$の点を __分離__ するとは任意の$$ x\neq y \in X$$に対し,ある$$f \in A$$が存在し,$$f(x) \neq f(y)$$となることとします。

$$A \subset C(X, \mathbb{R})$$が$$C(X, \mathbb{R})$$の部分代数は定義は書きませんが、和と積で閉じているものです。

実際に定理の主張を述べます。

|Theorem(StoneWeierstrassの定理)|
|:--|
|$$X$$がCompactな位相空間で,$$A \subset C(X, \mathbb{R})$$が$$C(X,\mathbb{R})$$の部分代数であり、$$X$$の点を分離するならば,$$A$$は稠密である.|||

非常に詳しく解説された日本語記事があるので、詳細はそこを参照してください。
http://integers.hatenablog.com/entry/2016/07/29/155652


では、実際にそれを使って、この論文で示す定理を証明したいと思います。

|Theorem 7|
|:--|
|Let $$f$$ be an $$\mathcal{F}$$-invariant function. For any positive integer N, there exists a degree N polynomial $$h_N$$ of $$hom(F, G)$$ s.t. $$f(G)\approx h_N (G)$$ $$\forall G$$ with $$\vert V (G) \vert \le N$$|||


- $$\mathcal{G}_N$$はノード数N以下のグラフ全体
- $$\mathcal{G}_N/ \mathcal{F}$$は$$\mathcal{F}$$-indistinguableが定める同値関係で割ったもの.
- $$\bar{G}$$は$$G$$の同値類

とします。

証明するべきことを考えると、$$f$$が$$\mathcal{F}$$-invariant functionの時,これが$$h_N(G)$$で近似できることを言えばよいです。

また,$$f$$-が$$\mathcal{F}$$-invariant なのでこの時,$$f$$は$$\mathcal{G}\_N/\mathcal{F}$$から$$\mathbb{R}$$への連続関数を誘導します。
さらに$$\mathcal{G}\_N/\mathcal{F}$$から$$\mathbb{R}$$への連続関数は$$\mathcal{F}$$-invariantな関数と一対一に対応するので、$$\mathcal{G}_N/ \mathcal{F}$$上で考えます。

つまり、上のStone-Weierstrassの定理の記号に合わせると

$$X = \mathcal{G}_N/ \mathcal{F}$$です。

ちなみに、$$X$$で考えるのは分離性のためです。

近似する側は
$$\mathcal{A}:= \\{f \in Map(\mathcal{G}\_N/ \mathcal{F}, \mathbb{R}) \mid \exists n \in \mathbb{Z}\_{\ge 0} \exists F\_1,\ldots F_n \in \mathcal{F} \exists h \in \mathbb{R}[X_1, \ldots, X_n] s.t.  f(\bar{G}) = h[hom(F_1, G), \ldots, hom(F_n, G)] \\}$$
です。

StoneWeierstrassの定理を使って示すために以下の三条件をチェックしましょう。

1. $$X$$のCompact性
2. $$\mathcal{A}$$が$$C(X, \mathbb{R})$$の部分代数になっていること
3. $$\mathcal{A}$$が分離性の仮定を満たすこと.

1はすぐ示されます。

$$\mathcal{G}_N/\mathcal{F}$$はノード数を制限していあるので、有限集合となり、離散位相でコンパクトになります。

2も計算すればわかります。$$\mathcal{A}$$が代数であることは
$$f_1 = h_1[hom(F_1, \cdot ), \ldots, hom(F_n, \cdot)]$$
$$f_2 = h_2[hom(F_{n+1}, \cdot ), \ldots, hom(F_m, \cdot)]$$に対し,
$$h_3(x_1, \ldots, x_m) = h_1(x_1\ldots, x_n) + h_2(x_{n+1} , \ldots, x_m)$$
$$h_4(x_1, \ldots, x_m) = h_1(x_1\ldots, x_n) \cdot h_2(x_{n+1} , \ldots, x_m)$$

とすると,

- $$(f_1 + f_2)(G)= h_3[hom(F_1, G), \ldots, hom(F_m, G)]$$
- $$(f_1 \cdot f_2)(G)= h_4[hom(F_1, G), \ldots, hom(F_m, G)]$$

とかけ、積や和について閉じていることがわかります。

3.の分離性ですが、全ての$$g \in \mathcal{A}$$に対し、$$g(G_1) = g(G_2)$$とすると,
$$g$$として$$g(G) = hom(F_i, G)$$を取ることにより,$$G_1$$と$$G_2$$は$$\mathcal{F}$$-indistinguableであり,
$$\mathcal{G}_N/\mathcal{F}$$上一致します。
よってこの対偶をとれば,分離性が示されます。

よって,StoneWeierstrassの定理より$$\mathcal{A}$$は$$C(X, \mathbb{R})$$上稠密であることがわかります。
また、どちらも有限次元$$\mathbb{R}$$-線形空間なので,一致します。

実際に次数$$N$$以下まで正確にBoundできるかはちょっとよくわかりませんでしたが、十分高い次数を選べばそれで表せることは言えると思います。

ここまでがこの定理の証明です。


### 一般のグラフ全体で示すためには
これはすぐわかるように次数でBoundされているところが課題になります。

そこで、一般の場合で示したいのですが、離散位相で適当に位相を入れてもうまく行きません。

そこでここではGraphonと呼ばれるものを使います。Graphonはグラフのある種の極限で表されるもの全体です。
Graphon全体がCompactであり、他2つの条件は上と同じように定めれば,分離的なので,証明できます。

実際には位相の議論(どういう位相で,limitとの交換性等があるのか)を気にするべきところですが、ここでは一旦省略します。


### 計算複雑性
- 実際にGraphを計算にどの程度時間がかかるかが述べられてます。
  グラフのtree decompositionを使うことで, $$hom(F,G)$$は$$O(|V(G)|^{tw(F) +1})$$でboundできるようです。
  $$\mathcal{F}$$がtreeの集合の場合は動的計画法で簡単に計算できるそうです。


## Graphs with Features

先程はグラフについて調べましたが、機械学習では各ノードにデータが付随した対象を考えます。まずはそれを定義しましょう。

|Definition|
|:--|
|vetex-featured graphとはグラフ$$G$$と$$x :V(G) \to \mathcal{X}$$, where $$\mathcal{X}= [0, 1]^p$$の組$$(G,x)$$である.|||

bijection $$\sigma: V(G) \to U$$に対し,$$x^{\sigma}:U \to \mathbb{R}, u \mapsto x(\sigma^{-1}(u))$$で定める.

$$G_1^{\sigma} = G_2, x_1^{\sigma} = x_2$$となる$$\sigma: V(G_1) \to V(G_2)$$が存在する時,$$(G_1, x_1)$$と$$(G_2, x_2)$$が同型と定める.


### Weighted Homomorphism Numbers
最初に重みが非負実数である場合に、重み付き homomorphism numberを以下で定義します。

$$
\mathrm{hom}(F, (G,x)) = \sum_{\pi \in \mathrm{Hom}(F,G)}\prod_{u \in V(F)} x(\pi(u))
$$

この場合にTwin Reductionを以下で定義する.
もし$$u, v \in G$$が近傍が一致するかどうかでノードごとに同値関係を定める

その時$$G' = G/ \sim$$とする.エッジについても同値関係で自然に誘導されるものとする.
$$x':G' \to \mathbb{R}$$は$$x(g')= \sum_{g \in g'}x(g)$$で定める.


__remark__
$$V(G'): = V(G) \setminus  \{u, v\} \cup  \{w\}$$
$$x':V(G') \to \mathbb{R}$$を

- $$g \in V(G)$$の時$$x'(g)= x(g)$$
- $$g = w$$の時$$x'(w) = x(u) + x(v)$$

で定めるプロセスを収束するまで繰り返すのでtwin reductionというのだと思います.

twin reductionを$$G^t$$で定める

この時、以下が成り立つようです。

|Theorem 13((Freedman et al., 2007), (Cai & Govorov, 2019)).|
|:--|
|$$(G_1, x_1)$$と$$(G_2, x_2)$$が$$G_1^t= G_1,G_2^t=G_2$$が成り立ち$$\forall v \in G_1, x_1(v) \neq 0, v' \in G_2, x_2(v) \neq 0$$する.この時$$(G_1, x_1)$$と$$(G_2, x_2)$$が同型であることは任意の単純グラフ$$F$$に対し,$$\mathrm{hom}(F, (G_1, x_1)) = \mathrm{hom}(F, (G_2, x_2))$$となることと同値|||

__remark__
$$G^{tt} = G^{t}$$となる.


今までは一次元非負でしたが、次は多次元の場合に考えます。
多次元の場合は$$x:V(G) \to [0, 1]^p$$で,さらに
$$\phi:\mathbb{R}^p \to \mathbb{R}$$を使い,
$$
\mathrm{hom}(F,G,x;\phi) = \sum_{\pi \in \mathrm{Hom}(F, G)} \prod_{u\in V(F)} \phi(x(\pi(u)))
$$
とします。これを$$(F, \phi)$$-convolutionといいます.

例えば$$\phi$$が$$1$$次元目へのprojectionとすると
$$\mathrm{hom}(F,G, x; \phi) = \mathrm{hom}(F, (G, x_1))$$となります.
また$$\phi$$が1へのconstatnt mapなら
$$\mathrm{hom}(F,G, x; \phi) = \mathrm{hom}(F, G)$$となります.


これについても同型に関係する定理(Theorem14)が記載されています。
ただし、内容がちょっとよくわからなかったので、論文のまま記載します.
__Theorem 14__
> Two graphs $$(G_1, x_1)$$ and $$(G_2, x_2)$$ are
isomorphic if and only if $$\mathrm{hom}(F, \phi,(G_1, x_1)) = \mathrm{hom}(F, \phi,(G_2, x_2))$$ for all simple graph $$F$$ and some continuous function $$\phi$$

証明等をここは言及せずに不明点を述べておきます.

- Statementに出てくる不明点
  - $$\mathrm{hom}(F, \phi,(G_1, x_1))$$が定義されていない.
    後ろに定義された$$\mathrm{hom}(F, G, x; \phi)$$と同じ?
  - $$\phi$$は存在でいいか?
    $$\phi$$が存在を意味する場合かつhomの定義が上のとおりとすると, $$\phi(x) = 0$$となる関数を取ると$$x: V(G) \to \mathbb{R}^p$$の値域さえ一致していれば,必ず一致するので,明らかに成り立たない.
- 証明での不明点
  - 上の定理13を使う箇所があるがtwin redunction後の仮定が必要なのでは?
  グラフが同型だが、$$x$$込みでは同型でなく,twin-reduction後なら同型になるものなんていくらでもある.例えば近傍が$$v$$のみの持つノード$$u_1,u_2$$に対し,$$x(u_1) + x(u_2) = y(u_1) + y(u_2)$$かつ$$x(u_1) \neq y(u_1)$$とすれば$$(G, x) \neq (G, y)$$になる.

この定理を以降で直接使うわけではないので、一旦気にせず、次にすすめることにします。

## Weighted Graphの普遍近似定理
Weighted Graphについても普遍近似定理を示しましょう。
今までと同様に写像を定義します。

$$(G_1, x_1)$$と$$(G_2, x_2)$$が$$(\mathcal{F}, \Phi)$$- __indistinguable__ を$$\mathrm{hom}(F, \phi,(G_1, x_1)) = \mathrm{hom}(F, \phi, (G_2, x_2))$$で定める.
$$f$$が$$(\mathcal{F}, \Phi)$$-__invariant__ とは$$(G_1,x_1), (G_2, x_2)$$が$$(\mathcal{F}, \Phi)$$-indistinguableな時,$$f(G_1, x_1) = f(G_2, x_2)$$となるものをいう.

$$\hom(\mathcal{F}, G, x; \Phi) = [\hom(F, G, x; \phi)\mid F \in \mathcal{F}, \phi \in \Phi]$$とする.これを$$(\mathcal{F}, \Phi)$$-convolutionといいます.

$$(\mathcal{F},\Phi)$$-indistinguableが定める同値関係で割った空間を$$\tilde{\mathcal{G}}$$とします。
これは$$\mathcal{G}/ \mathcal{F}^{\Phi}$$と全単射になると書いてあったのですが、これもよくわかりませんでした。

極端な例を上げると$$\mathcal{F}$$を一点からなるグラフのみ,$$\Phi=  \{0\}$$つまり,0写像だけの集合とすると,$$\mathrm{hom}(F,G, x; 0) =0$$より,
$$\tilde{\mathcal{G}}$$は一点集合となる.
一方で$$\mathcal{G}/ \mathcal{F}$$は明らかに一点でないためこれは同型ではありません.

正当化するなら$$\mathcal{G}$$は$$(G, x)$$全体の集合とし,
ただし$$x: V(G) \to [0, 1]$$で,
今直積の各成分の$$\mathcal{G}/\mathcal{F}$$は$$(G, x)$$を$$\mathrm{hom}(F, (G,\phi \circ x))$$が一致しているかどうかで割った空間を表す.
つまり$$\mathcal{G}/ \mathcal{F}$$の直積に見えるが,各成分は$$\phi$$での情報を含めて割られています.

この場合は$$\tilde{G} \to \mathcal{G}/\mathcal{F}^{\Phi}, (G, x) \mapsto (G, \phi(x))_{\phi}$$は,well-definedかつ,単射になります。
全射性は確認できませんでしだが,成り立つと思うことにします.

その下にある

> Each coordinate of the |Φ|-dimensional
space is completed to a compact Hausdorff space (Borgs
et al., 2008). Therefore, by the Tychonoff product theorem (Hart et al., 2003), the |Φ|-dimensional space is compact.

もどういう意味なのかわかりかねたのですが,

- GraphonはWeighted Graphの場合も含みGraphonはCompact(もしかしたらGraphonでない正当化をしているかもしれません)
- CompactのQuotientはCompact
- 直積もチコノフからCompact

という意味で全体がCompactになることを示しているのだと思います.

これから前と同様にして

|Theorem|
|:--|
|$$(\mathcal{F}, \Phi)$$-invariant continuous funcitionは $$hom(F, G, x;\phi)$$が生成する代数で稠密である.|||

ことが言えます、(これも本当は位相の議論は気にするべきところです.)


さらにここで
$$\mathcal{F}, \Phi$$-convergent という概念を定義します.
$$(G_i, x_i)$$が$$(\mathcal{F}, \Phi)$$-convergentとは,$$\mathrm{hom}(F, G_i, x_i,; \phi)$$が収束することで定めます.
さらに$$(\mathcal{F}, \Phi)$$-continuous functionを$$(\mathcal{F}, \Phi)$$-convergentなweighted graphの列$$(G_i, x_i)$$に対し,
$$\lim_{i \to \infty} f(G_i, x_i)$$が存在し,さらにその値が$$\mathrm{hom}(F, G_i,x_i, \phi)$$のlimitのみによることをいいます.

これについても普遍近似性を示します.
$$C(\mathcal{G}; \mathcal{F}, \Phi)$$は$$(\mathcal{F}, \Phi)$$-continuous funcitonとします.

$$\mathcal{H} \subset Hom([0, 1]^p, \mathbb{R})$$をコンパクト開位相で位相空間としたときの稠密な集合とする。
例えば多項式が定める写像全体や,NN(活性化関数によっては示されていない可能性があることに注意)がこれに該当します。


次に
$$
\mathcal{H}(\mathcal{G}; \mathcal{F}, \Phi) = \{\sum_{F \in \mathcal{F}, \phi \in \Phi} h_{F, \phi}(\operatorname{hom}(F, \cdot ; \phi): h_{F, \phi} \in \mathcal{H} \}.
$$
とする.($$\sum$$は非加算無限和を取りたいわけではないと思うのでやりたいのはあの中から有限個の$$F, \phi$$を選んで(有限個を除いて,$$h_{F, \phi}=0$$)の和にします.)
($$\mathcal{H}(\mathcal{G}; \mathcal{F}, \Phi) \subset C(\mathcal{G}; \mathcal{F}, \phi)$$)となる.


|Lemma 19|
|:--|
|$$\mathcal{F}$$を単純グラフ全体の集合とする. $$\Phi$$を$$[0, 1]^p$$から$$[0, 1]$$への連続写像全体とする.この時, $$(G, x) \mapsto \mathrm{hom}(\mathcal{F}, G, x; \Phi)$$は単射となる.|||
証明ですが,
1. $$G,G'$$が同型出ない時,異なる元に行くこと
2. $$G,G'$$が同型で$$x, x'$$が異なる時,異なる元い行くこと
の順番に示します.

1.の証明
$$(G, x)$$と$$(G',y)$$をweighted graphとする.
$$G$$と$$G'$$は同型でないグラフとすると
$$\phi=1$$を取ることにより,feaure lessの場合と同じ状況になるので,異なることが言える.


2.の証明
$$G \sim G'$$とする
今$$V(G) = \\{1, \ldots, n\\}$$とする.
さらにうまく置換することで,$$x(1) \le x(2) \le \ldots$$とできる.
同様にある置換$$\pi: G \to G$$で
$$y(\pi(1)) \le y(\pi(2)) ...$$とできるものがある.
同型でない場合,$$x(u) \neq y(u)$$となる$$u$$が存在する.今$$y(u) \ge x(u)$$とする.
今こうした最小の$$u$$に対し
$$\phi(x)$$を$$x \le x(u)$$の時1, そうでない時$$0$$とする.
(これは不連続だが,$$x(u),y(u)$$は有限個なので,これに対して同じ値を取る連続関数は存在する.)
この時,
$$\pi(v)= u, \pi(F) \subset \\{1, \ldots, u\\}$$となる$$\pi \in Hom(F, G)$$が存在したら,
$$\prod_{u \in V(F)} \phi(x(\pi(u)) > \prod_{u \in V(F)} \phi(y(\pi(u))$$となり,他も不等号が成り立つので,

$$\mathrm{hom}(F, G, x; \phi) \neq \mathrm{hom}(F, G, y; \phi)$$

が示される.
実際$$F$$として一点を取ることでこのような写像の存在も言えるので、単射であることが言えた.



$$(\Phi, \mathcal{F})$$はlemma 19と同じ条件とする.

|Theorem 20|
|:--|
|$$\mathcal{G}$$はノードの数が有界な,graphの作るコンパクト集合とする. この時 $$\mathcal{H}(\mathcal{G}; \mathcal{F}, \Phi)$$はdenseとなる.|||

今の場合ノードを有限にしているので、これらの定めるGraphonはグラフで表現できる.この時参考文献では,これがCompactになることまで示されているようには思わなかったが,ここはCompactになっているとする.(階段の個数が有限までの階段関数の集まりなので、示すことはできるはず)


この時,単射性から$$(G_1, x_1) \neq (G_2, x_2)$$の時,$$hom(\mathcal{F}, G_1, x; \Phi) \neq \hom(\mathcal{F}, G_2, x_2; \Phi)$$となり, この二点で異なる$$h \in \mathcal{H}$$は稠密性から存在するので,$$\mathcal{H}(\mathcal{G}; \mathcal{F}, \Phi)$$が部分代数であれば,定理20は成り立つ.

__remark__
$$\mathcal{H}$$がdenseなだけでなく,積で閉じないといけないとStone Weierstrassの仮定を満たさないと行けない気が...


## 実装する場合
ここで例をあげられているのは

- 次数6までの木全体
- 次数8までのサイクル全体

等であった.

これでGINで学習できないものが学習できたようです.

このぐらいのグラフで十分精度が出るのは面白いなと感じました.

## 所感

感想ですが、
- グラフ理論に対する知識を使ってNNを解析してるのが、新鮮で面白なと感じました
- 実験結果としてもそこまで複雑でないグラフで高い精度が実現できていことに驚きました。
- 数学的な部分はもう少し補足がないと厳しいなと感じるところがちらほらありますが、ページ制約的にも査読的にも機械学習論文ではしょうがないですし、補足できるものが多く、論文自体の完成度が高いなと感じました。


## GraphonのRemark
最後にGraphonについて少し定義と基本的な性質を述べておこうと思います.

https://arxiv.org/pdf/math/0702004.pdf
を参考にしました。

$$\mathcal{W}$$を有界で対称的な可測関数$$W:[0, 1]^2 \to \mathbb{R}$$全体の集合とする.
$$I \subset \mathbb{R}$$をbounded intervalに対し
$$W \in \mathcal{W}$$かつ$$W(x, y) \in I$$となる関数を$$\mathcal{W}_I$$で表す.$$\mathcal{W}$$の元をGraphonという.


$$[0, 1]$$の分割$$P$$がmeasurerableとは$$p \in P$$がルベーグ可測という意味である.
$$\forall p, p' \in P, m(p)= m(p')$$となる時,equipartitionという.

$$W \in \mathcal{W}$$がstep functionとは,ある[0, 1]の分割$$ \{V_1, \ldots, V_K\}$$が存在し$$W$$が$$V_i \times V_J$$上constatnになるものである.


特に$$V_i$$が全てintervalとして取れる時interval step functionという.


グラフ$$G$$に対し,その隣接行列を$$A$$とする.$$A$$が$$n \times n$$行列のとき,$$[i/n, i+1/n] \times [j/n, j+1/n]$$上では$$A\_{ij}$$の値を取る関数でグラフを表現できる.

Graphonの空間はComapct Hausdorffであることが知られており,グラフ全体を自然に含んでいる.


また重要な結果として,以下がある.

|Theorem(Theorem 3.8の一部)|
|:--|
|$$I$$はfinite interval.$$(W_n)$$はgraphonの列とする.この時以下は同値. <br> $$\bullet$$ $$t(F, W\_n)$$は収束する. <br> $$\bullet$$ metric$$\delta$$で$$W_n$$はコーシー列となる. |||

|Theorem(Corollary 3.9)|
|:--|
|For any convergent sequence $$(G_n)$$ of weighted graphs with uniformly bounded edgeweights there exists a graphon W such that $$\delta(W_{G_n}, W) \to 0$$. Conversely, any graphon $$W$$ can be obtained as the limit of a sequence of weighted graphs with uniformly bounded edgeweights. The limit of a convergent graph sequence is essentially unique: If Gn → W, then also Gn → W′ for precisely those graphons W′ for which $$\delta(W, W′) = 0$$.|



よって$$t$$の意味での極限と思うことができ,論文ではそちらで扱っている.(この場合もWeighted Graphでないといけないはずだが)
