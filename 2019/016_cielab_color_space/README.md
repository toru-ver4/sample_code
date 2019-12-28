# CIELAB色空間の基本的なところを確認

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
