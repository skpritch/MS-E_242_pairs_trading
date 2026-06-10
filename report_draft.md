\documentclass[12pt]{amsart}
\usepackage{amssymb, amsthm}
\usepackage{mathtools}
\usepackage{mathrsfs}
\usepackage{hyperref}
\usepackage{fullpage}
\usepackage{soul}
\usepackage{cleveref}
\usepackage{tabularx}
\usepackage{hyperref}
\usepackage{float}
\usepackage{tikz}
\usepackage{bbm}
\usepackage{listings}
\usepackage{booktabs} % For professional table formatting


\lstset{
  basicstyle=\ttfamily\small,  % Use a monospaced font
  breaklines=true,             % Automatically wrap long lines
  frame=single,                % Add a frame around the code
  numbers=left,                % Line numbers on the left
  numberstyle=\tiny,           % Line number font size
  captionpos=b,                % Position of captions (bottom)
  tabsize=4                    % Tab width
}

\usetikzlibrary{decorations}
\usetikzlibrary{snakes}
\usetikzlibrary{decorations.pathreplacing,calligraphy}
 \floatplacement{figure}{H}
\hypersetup{
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    pdftitle={Overleaf Example},
    pdfpagemode=FullScreen,
    }

\urlstyle{same}

\newtheorem*{theorem*}{Theorem}
\newtheorem{theorem}{\textbf{Theorem}}[section]
\newtheorem{proposition}[theorem]{\textbf{Proposition}}
\newtheorem{corollary}[theorem]{\textbf{Corollary}}
\newtheorem{lemma}[theorem]{\textbf{Lemma}}
\newcommand{\bhat}[1]{\hat{\Bar{#1}}}
\newcommand{\aln}[1]{\begin{align*}#1\end{align*}}
\newcommand{\half}{\frac{1}{2}}
\newcommand{\RR}{\mathbb{R}}
\DeclareMathOperator*{\argmax}{arg\,max}
\DeclareMathOperator*{\argmin}{arg\,min}
%\newtheorem{conjecture}[theorem]{Conjecture}

\theoremstyle{definition}
\newtheorem{definition}[theorem]{Definition}
\newtheorem{remark}[theorem]{Remark}
\newtheorem{example}[theorem]{Example}
\newtheorem{assumption}[theorem]{Assumption}

\numberwithin{equation}{section}

\title{Embeddings-Based Pairs Trading}
\author{Simon Pritchard, Elijah Schacter, Daniel Yang}
\date{March 2025}

\begin{document}
\maketitle

\section{Problem Description and Motivation}

\section{Data} We use 30-minute bars with open, high, low, close and volume data for US equities from June 2016 through December 2025 (63M rows). We restrict to regular trading hours (09:30–16:00 ET) and use the bar close as the price. The raw universe contains 1,749 symbols and good coverage: 1,426 of 1,749 have at least 95\% non-missing bars. We align all symbols onto a common 30-minute regular-hours grid and compute log returns as $r_t = log(\frac{p_t}{p_{t-1}})$. To avoid unreasonable trading assumptions on illiquid names, each walk-forward window selects a liquidity-filtered universe: the top-150 symbols by median per-bar dollar volume (we approximate this as close x volume) computed on the training window only. Within each window we further require at least 95\% non-missing bars. Any bars that are missing are assumed to have no trades and their values are assigned accordingly. 

\section{Exploratory Analysis and Pre-Experiment Setup} We did three pieces of exploratory analysis: Establish a clear fixed trading rule, build a basic pairs-trading baseline via correlated pairs, and use linear autoregression to determine the best lag periods to input into our later ML models in each of the rolling training windows.

\subsection{Trading Strategy} Before we construct the basic pairs-trading baseline we establish a clear trading rule that will be maintained across all future experiments. This keeps the results as a clear treatment vs. control comparison of traditional pairs selection vs. our embeddings-based approach. Given a selected pair (a,b), we fit a hedge ratio $\beta$ and intercept $\alpha$ by OLS of $log(p_a)$ on $log(p_b)$ in the training window: $log(p_a) = \alpha + \beta \cdot log(p_b)$, following the standard cointegration regression of Avellaneda and Lee \cite{avellaneda2010}. From there we pull out the residual to be the spread $s_t = log(p_{a,t})- \alpha -\beta \cdot log (p_{b,t})$. We compute the z-score using a causal rolling window whose length is tied to the pair's Ornstein--Uhlenbeck half-life: $w = \max(\lceil 2\,h \rceil,\, 5)$ bars, where $h$ is the OU half-life estimated on the training spread. The rolling z-score is $z_t = (s_t - \bar{s}_{t-w:t})\,/\,\hat\sigma_{t-w:t}$, using only past data at each bar. We enter when $|z_t|>2.5$ (short a and long $\beta \cdot b$ when $z_t$ positive and the reverse when $z_t$ is negative), exit when $|z_t|<0.5$, and set a stop at $|z_t|>4$.
\subsection{Thresholds}
We also create two more thresholds that a pair must satisfy to be eligible to be traded. First, we set a volatility floor at $\sigma>0.005$ FIND REFERENCE FOR THIS. The reasoning behind this is that if the spread has tiny volatility the actual expected price spread will be small too; we don't want to trade on these because they will be eaten up by trading frictions/costs. Second, we run a simple mean Breusch-Pagan test to make sure that the residuals are homoskedastic. If they are not, pairs trading based on a z-score that assumes $\sigma$ is stationary does not make sense. The test is:
\[
s_{(t,i)}^2 =\alpha_0 + \alpha_1\cdot log(p_b)
\]
If $\alpha_1\neq 0$ at the 95\% confidence interval we do not include the pair. Third, we require the spread's Ornstein--Uhlenbeck half-life to lie between 1 and 150 bars. The half-life is estimated on the training spread via the AR(1) discretization $\Delta s_t = a + b\,s_{t-1} + \varepsilon_t$; the half-life is $-\ln 2 / \ln(1+b)$. Pairs with $b\geq 0$ (non-mean-reverting) are rejected outright, and those reverting too fast ($<$1 bar, likely microstructure noise) or too slowly ($>$150 bars, capital-inefficient) are excluded. The test-time trading strategy only trade on the top 20 pairs in the training period by Sharpe ratio, constantly weight-adjusted equally across active pairs so that gross exposure = 1. Transaction costs of 5 bps/side (10 bps round-trip) are charged on each traded leg. All parameters are estimated on training data and frozen for the test window. 

\subsection{Classic pairs trading baseline} The classic pairs trading baseline uses the top 300-pairs in pair-wise return correlation as the trading universe. In addition to this, a hard threshold $\rho>0.5$ is applied. We then use the above trading strategy to evaluate performance.  

\section{Model Description} We implement two encoding-based models to select the best pairs to trade. Each walk-forward window uses a training period of 252 trading days ($\sim$1 year) and a test period of 63 trading days ($\sim$1 quarter), rolled quarterly.

\subsection{Autoencoder} The auto-encoder's input and output are 12-dimensional thirty-minute log-returns of each stock in each day of the training set. The autoencoder contains hidden MLP layers of shape 64,16,64. There are 37,800 training data points $\sim$ 252 training days x 150 stocks. For each stock, we then do 252 (total days) forward passes up to the 16-dimensional hidden layer. Lastly, we take the average of the normalized 16-dimensional layers across all 252 days and use that as the encoder for each stock. From there we use the 8 nearest neighbors as the possible pair candidates per stock. Then, similar to the classing pairs trading approach, we select the top 300 pairs by neighbor distance and implement the trading strategy defined in section 3.1.

\subsection{Linear autoregression} Our autoregressive model is done stock-by-stock in the chosen top-150 symbols in each of the training windows. Critically, information about autoregressive parameters are kept localized to each window; this is necessary to prevent issues regarding looking into the future. The model is simple:
\[
r_t = \sum^{15}_{i=1}r_{t-i}
\]
In short, we autoregress a stock's returns using its past 15 returns. Our goal here is to figure out where we want to place the lag cutoff for the GRU model described below. We do not base this test using significance because it is not valid to make a clean comparison between significant in linear space and non-linear space. Instead we set a hard cutoff; going backwards in time, we stop when the first lag term that we see with a coefficient less than $0.01$ appears. The resulting per-stock lookback is then used directly in the GRU model.

\subsection{Distinct Recursive ML} We implement a simple recurrent embedding model using time-lagged log returns. The lag for each stock is determined by the per-stock lookback established in the linear autoregression step above. We then train one shared GRU model across all stocks in the liquidity-filtered universe. This ensures that the hidden vectors for different stocks lie in the same latent space and can be compared using nearest-neighbor distance.

Each training example consists of a rolling window of lagged log returns for one stock, and the target is that stock's log return in the next timestep. The architecture uses a GRU layer with hidden dimension 64 as the recursive base, followed by a one-layer MLP with a 16-dimensional hidden layer to predict the next log return. Since each stock contributes around (252 x 12) training examples per walk-forward window, the shared model is trained on the pooled set of stock-window observations rather than on each stock independently.

After training, we run each stock's training-window sequences through the shared GRU-MLP model and extract the 16-dimensional hidden vector immediately before the final output layer. We average the normalized hidden vectors across the stock's training examples to obtain one latent representation per stock. As with the autoencoder approach, we then use the 8 nearest neighbors in this latent space to form candidate pairs and implement the fixed trading approach from Section 3.1.

\section{Training, Validation, Testing}

\section{Parameter/Hyperparameter Selection}

\section{Performance Evaluation}

\section{Results}

\section{Limitations and Conclusion}

\section{Disclosures} Data Source: Polygon/Massive. Packages: Standard Python packages (torch, numpy, pandas, etc). AI tools: Claude Code and Codex (mainly used for pulling data, and making plots pretty).

\end{document}