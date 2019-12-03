# CIELAB色空間の基本的なところを確認

## 変換式

$$
\begin{array}{ll}
L* = 116 f\displaystyle\left(\frac{Y}{Y_n}\right) \\
a^* = 500 \displaystyle\left(f\left(\frac{X}{X_n}\right) - f\left(\frac{Y}{Y_n}\right)\right) \\
b^* = 500 \displaystyle\left(f\left(\frac{Y}{Y_n}\right) - f\left(\frac{Z}{Z_n}\right)\right)
\end{array}
$$


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
