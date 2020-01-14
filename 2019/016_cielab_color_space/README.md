# CIELAB色空間で BT.2020色域の Gamut Boundary をプロットする

## はじめに

この記事は「Report ITU-R BT.2407 の Annex2 を実装」シリーズの2回目です。前回は xyY色空間でしたが、今回からは CIELAB色空間の話になります。

## 背景

BT.2407 の Annex2(以後、BT.2407と略す) の実装には、CIELAB色空間における BT.2020色域 および BT.709色域の Gamut Bounary の算出が必要である。前回の記事では事前確認として xyY色空間で Gamut Boundary を算出したが、今回は本命の CIELAB色空間で Gamut Boundary を算出することにした。

## 目的

* CIELAB色空間で BT.2020色域の Gamut Boundary を算出してプロットする。

## 結論


## 算出方法

### 方針

基本的な方針は前回の記事と同様である。以下の動画のように a\*b\* 平面を L\* 方向に積み重ねて CIELAB空間を表現することで Gamut Boundary を算出することにした。

---

ここに動画

---

したがって、解くべき問題は **CIELAB色空間における 任意の L\* に対する a\*b\*平面の Gamut Boundary の算出** となった。言い換えると、上の動画の左画面 の Gamut Boundary を算出するのが目的である。

### 理論

前回の記事と同様に CIELAB色空間の単純な特性を利用して a\*b\* 平面の Gamut Boundary を求めることにした。その特性とは **CIELAB色空間の Gamut Boundary の外側の L\*a\*b\* 値は RGB値へ変換すると 0.0～1.0 の範囲を超える** である。よって詳細は後述するが 任意の

### 定式化

前項で述べた性質を利用するためには L\*a\*b\* to RGB 変換が必要である。ここでは変換式の定式化を行う。

まず、XYZ to L\*a\*b\* の計算式を確認する。計算式は以下の通りである[1][2]。

$$
\begin{array}{ll}
L^{*} &= 116 f\displaystyle\left(\frac{Y}{Y_n}\right) -16 \\
a^* &= 500 \displaystyle\left(f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right) \\
b^* &= 500 \displaystyle\left(f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right)\right)
\end{array}
$$

ここで $X_n, Y_n, Z_n$ は白色点の XYZ値を意味する。また $f(t)$ は以下の通りに定義される。

$$
f(t) = \left\{
\begin{array}{ll}
\displaystyle\ t^{1/3} & \text{if } t > \sigma ^3 \\
\displaystyle\frac{t}{3\sigma ^2} + 4/29 & \text{otherwise}
\end{array}
\right.
$$

なお、$\sigma = 6/29$ である。

これの逆関数を計算すると L\*a\*b\* to XYZ の計算式が得られる。結果は以下の通りである。

$$
\begin{array}{ll}
X = X_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116} + \frac{a^*}{500}\right) \\
Y = Y_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116}\right) \\
Z = Z_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116} - \frac{b^*}{200}\right)
\end{array}
$$

$$
\begin{array}{ll}
f^{-1}(t) = \left\{
\begin{array}{ll}
t^3  & \text{if } t > \sigma \\
3 \delta^{2} \displaystyle\left(t - \frac{4}{29}\right) & \text{otherwise}
\end{array}
\right.
\end{array}
$$

さて、続いて XYZ to RGB 変換の数式を組みたいところだが、ここで少し式の表現方法を変える。なぜならば X, Y, Z の数式はそれぞれ $f^{-1}(t)$ が2種類存在するからである。以下に示すように X, Y, Z を定義しなおす。

まず、X, Y, Z の $f^{-1}(t)$ の $t$ に該当する箇所をそれぞれ $t_X, t_Y, t_Z$ と置く。

$$
\begin{array}{ll}
X = X_n f^{-1} \displaystyle\left(t_X \right) \\
Y = Y_n f^{-1} \displaystyle\left(t_Y \right) \\
Z = Z_n f^{-1} \displaystyle\left(t_Z \right)
\end{array}
$$

続いて新たに変数 $i, j, k$ を設けて $t_X, t_Y, t_Z$ の値に応じて場合分けする。

$$
\begin{array}{ll}
X_i = \left\{
\begin{array}{ll}
X_n t_X^3 & \text{if } i=0, t > \sigma \\
X_n 3 \sigma ^2(t_X - \frac{4}{29}) & \text{if } i=1, t \leq \sigma \\
\end{array}
\right. \\
Y_j = \left\{
\begin{array}{ll}
Y_n t_Y^3  & \text{if } j=0, t > \sigma \\
Y_n 3 \sigma ^2(t_Y - \frac{4}{29}) & \text{if } j=1, t \leq \sigma \\
\end{array}
\right. \\
Z_k = \left\{
\begin{array}{ll}
Z_n t_Z^3  & \text{if } k=0, t > \sigma \\
Z_n 3 \sigma ^2(t_Z - \frac{4}{29}) & \text{if} k=1, t \leq \sigma\\
\end{array}
\right. \\
\end{array}
$$

ここで、XYZ to RGB 変換の行列 $M$ を以下の通りに定義する。

$$
M = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
$$

すると、最終的に L\*a\*b\* to RGB の計算式は以下となる。

$$
\begin{array}{ll}
R_{ijk} = a_{11}X_i + a_{12}Y_j + a_{13}Z_k \\
G_{ijk} = a_{21}X_i + a_{22}Y_j + a_{23}Z_k \\
B_{ijk} = a_{31}X_i + a_{32}Y_j + a_{33}Z_k \\
\end{array}
$$

なお、式を見れば分かる通り RGB値はそれぞれ $i, j, k$ の組み合わせにより **8通り** 存在するので注意が必要である。

## 参考文献

[1] ISO/CIE, "Colorimetry — Part 4: CIE 1976 L*a*b* colour space", 2019.

[2] Wikipedia, "CIELAB color space", https://en.wikipedia.org/wiki/CIELAB_color_space















## 変換式

まずは $XYZ$ to $L^{*}a^{*}b^{*}$ の変換式を確認する。

$$
\begin{array}{ll}
L^{*} &= 116 f\displaystyle\left(\frac{Y}{Y_n}\right) -16 \\
a^* &= 500 \displaystyle\left(f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right) \\
b^* &= 500 \displaystyle\left(f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right)\right)
\end{array}
$$

ここから逆関数を求める。まず、$Y$ について式を解く。

$$
\begin{array}{ll}
L^{*} &= 116 f\displaystyle\left(\frac{Y}{Y_n}\right) -16 \\
\displaystyle\frac{L^{*} + 16}{116} &= f\displaystyle\left(\frac{Y}{Y_n}\right) \\
f^{-1}\left(\displaystyle\frac{L^{*} + 16}{116}\right) &= \displaystyle\frac{Y}{Y_n} \\
Y_n f^{-1}\left(\displaystyle\frac{L^{*} + 16}{116}\right) &= Y
\end{array}
$$

続いて、$X$ について解く。

$$
\begin{array}{ll}
a^* &= 500 \displaystyle\left(f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right) \\
\displaystyle\frac{a*}{500} &= \displaystyle f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right) \\
\displaystyle\frac{a*}{500} &= \displaystyle f\left(\frac{X}{X_n}\right) - \displaystyle\frac{L^{*} + 16}{116} \\
\displaystyle\frac{L^{*} + 16}{116} + \displaystyle\frac{a*}{500} &= \displaystyle f\left(\frac{X}{X_n}\right) \\
f^{-1}\displaystyle \left( \displaystyle\frac{L^{*} + 16}{116} + \displaystyle\frac{a*}{500}  \right) &= \displaystyle \frac{X}{X_n} \\
X_n f^{-1}\displaystyle \left( \displaystyle\frac{L^{*} + 16}{116} + \displaystyle\frac{a*}{500} \right) & = X
\end{array}
$$



関係ない

$$
\begin{array}{ll}
X = X_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116} + \frac{a^*}{500}\right) \\
Y = Y_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116}\right) \\
Z = Z_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116} - \frac{b^*}{200}\right)
\end{array}
$$


$$
\begin{array}{ll}
f^{-1}(t) = \left\{
\begin{array}{ll}
t^3 \\
3 \delta^{2} \displaystyle\left(t - \frac{4}{29}\right)
\end{array}
\right.
\end{array}
$$

chroma を $C^*$ とおくと、色相(Hue) $h^\circ$ を使って

$$
\displaystyle
\begin{array}{ll}
a^* = C^*\cos{h^\circ} \\
b^* = C^*\sin{h^\circ} \\
\end{array}
$$

という関係を導き出せる。これを元の式に代入すると

$$
\displaystyle
\begin{array}{ll}
X = X_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116} + \frac{C^*\cos{h^\circ}}{500}\right) \\
Y = Y_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116}\right) \\
Z = Z_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116} - \frac{C^*\sin{h^\circ}}{200}\right)
\end{array}
$$

となる。

ここで、更に XYZ信号を RGB信号に変換することを考える。変換行列を

$$
M = \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
$$

と置くと、

$$
\begin{array}{ll}
\begin{bmatrix}
R \\
G \\
B
\end{bmatrix}
&= \begin{bmatrix}
a_{11} & a_{12} & a_{13} \\
a_{21} & a_{22} & a_{23} \\
a_{31} & a_{32} & a_{33} \\
\end{bmatrix}
\begin{bmatrix}
X \\
Y \\
Z
\end{bmatrix} \\
&= \begin{bmatrix}
X a_{11} + Y a_{12} + Z a_{13} \\
X a_{21} + Y a_{22} + Z a_{23} \\
X a_{31} + Y a_{32} + Z a_{33} \\
\end{bmatrix}
\end{array}
$$

である。

実数解が欲しい場合は以下のように、`symbols` で `real=True` を指定すること。

```python
x = symbols('x', real=True)
y = (x - 1) ** 2 + 1
solution = solve(y)
print(solution)
```

$$
- 0.35569071615638 \left(\frac{l}{116} + \frac{4}{29}\right)^{3} - 0.275906005150939 \left(- \frac{c \sin{\left(h \right)}}{200} + \frac{l}{116} + \frac{4}{29}\right)^{3} + 1.63159672130732 \left(\frac{c \cos{\left(h \right)}}{500} + \frac{l}{116} + \frac{4}{29}\right)^{3} \\
1.61642498797091 \left(\frac{l}{116} + \frac{4}{29}\right)^{3} + 0.0171697720597756 \left(- \frac{c \sin{\left(h \right)}}{200} + \frac{l}{116} + \frac{4}{29}\right)^{3} - 0.633594760030686 \left(\frac{c \cos{\left(h \right)}}{500} + \frac{l}{116} + \frac{4}{29}\right)^{3} \\
- 0.0427770128712603 \left(\frac{l}{116} + \frac{4}{29}\right)^{3} + 1.02600958388031 \left(- \frac{c \sin{\left(h \right)}}{200} + \frac{l}{116} + \frac{4}{29}\right)^{3} + 0.0167674289909477 \left(\frac{c \cos{\left(h \right)}}{500} + \frac{l}{116} + \frac{4}{29}\right)^{3} \\
0.000177157244686691 c \sin{\left(h \right)} + 0.000419054568136006 c \cos{\left(h \right)} + 0.00110705645987945 l \\
- 1.10245861025907 \cdot 10^{-5} c \sin{\left(h \right)} - 0.000162730639912756 c \cos{\left(h \right)} + 0.00110705645987945 l \\
- 0.0006587933118851 c \sin{\left(h \right)} + 4.3064978145597 \cdot 10^{-6} c \cos{\left(h \right)} + 0.00110705645987945 l
$$

↑ボツ

## もう一度整理しよう。

$$
\begin{array}{ll}
Y = Y_n f^{-1} \displaystyle\left(\frac{L^* + 16}{116}\right) \\
\end{array}
$$

例えば、$L^* = 50$ の場合、$Y = 0.184$ である。RGB to XYZ 変換係数より以下が成立する。

$$
\begin{array}{ll}
Y &= a_{21} R + a_{22} G + a_{23} B \\
0.184 &= a_{21} R + a_{22} G + a_{23} B
\end{array}
$$

一方で

$$
\begin{array}{ll}
X &=  a_{11} R + a_{12} G + a_{13} B \\
Z &=  a_{31} R + a_{32} G + a_{33} B
\end{array}

$$

