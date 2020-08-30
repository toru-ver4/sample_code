# Formula

$$
\begin{aligned}
  \begin{bmatrix} R' \\ G' \\ B' \end{bmatrix} &=
     \begin{bmatrix}
      & &  \\
      & M_{CTM\_RGB} & \\
      & &  \\
    \end{bmatrix}
    \begin{bmatrix} R \\ G \\ B \end{bmatrix} \\

  \begin{bmatrix} X' \\ Y' \\ Z' \end{bmatrix} &=
     \begin{bmatrix}
       & & \\
      & M_{CTM\_XYZ} & \\
       & &
    \end{bmatrix}
    \begin{bmatrix} X \\ Y \\ Z \end{bmatrix} \\
  \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
  \end{bmatrix}
  \begin{bmatrix} R' \\ G' \\ B' \end{bmatrix} &=
   \begin{bmatrix}
     & & \\
     & M_{CTM\_XYZ} & \\
     & &
    \end{bmatrix}
  \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
  \end{bmatrix}
    \begin{bmatrix} R \\ G \\ B \end{bmatrix} \\
  \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
  \end{bmatrix}
  \begin{bmatrix}
    & & \\
    & M_{CTM\_RGB} & \\
    & &
  \end{bmatrix}
  \begin{bmatrix} R \\ G \\ B \end{bmatrix} &=
    \begin{bmatrix}
     & & \\
     & M_{CTM\_XYZ} & \\
     & &
    \end{bmatrix}
    \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
    \end{bmatrix}
  \begin{bmatrix} R \\ G \\ B \end{bmatrix} \\
  \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
  \end{bmatrix}
  \begin{bmatrix}
    & & \\
    & M_{CTM\_RGB} & \\
    & &
  \end{bmatrix} &=
    \begin{bmatrix}
     & & \\
     & M_{CTM\_XYZ} & \\
     & &
    \end{bmatrix}
    \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
  \end{bmatrix} \\
  \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
  \end{bmatrix}
  \begin{bmatrix}
    & & \\
    & M_{CTM\_RGB} & \\
    & &
  \end{bmatrix}
  \begin{bmatrix}
    & & \\
    & M_{RGB\_XYZ} & \\
    & &
  \end{bmatrix}^{-1} &=
  \begin{bmatrix}
    & & \\
    & M_{CTM\_XYZ} & \\
    & &
  \end{bmatrix} \\
  \begin{bmatrix}
    & & \\
    & M_{CTM\_XYZ} & \\
    & &
  \end{bmatrix} &=
\left[\begin{matrix}1.0 - 1.985 \alpha & 1.158 \alpha - 6.939 \cdot 10^{-18} & 0.6696 \alpha\\1.068 \alpha + 5.399 \cdot 10^{-17} & 1.0 - 1.782 \alpha & 0.7045 \alpha + 6.939 \cdot 10^{-18}\\1.163 \alpha + 3.469 \cdot 10^{-18} & 1.327 \alpha & 1.0 - 2.233 \alpha\end{matrix}\right]
\end{aligned}
$$
