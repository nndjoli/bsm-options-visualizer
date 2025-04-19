import models
import pandas as pd
import numpy as np
import yfinance as yf

NoneType = type(None)


def check_types(variable_name, value, types):
    if type(value) not in types:
        raise ValueError(
            f"The {variable_name} parameter must be of type {', '.join([type.__name__ for type in types])}."
        )


def check_s(s):

    ticker = None
    data = None

    if isinstance(s, str):
        ticker = yf.Ticker(s)
        data = ticker.history(period="1y", interval="1d")
        if data.empty:
            raise ValueError(
                f"The ticker {s} does not exist or has no data available."
            )
        else:
            s = ticker.history(period="1d", interval="1m")["Close"].iloc[-1]

    elif isinstance(s, (int, float)):
        if s <= 0:
            raise ValueError("The stock price must be a non-negative number.")
    else:
        raise ValueError(
            "The stock price must be a string (ticker) or a non-negative number."
        )

    return {"ticker": ticker, "data": data, "s": s}


def check_k(k):
    if isinstance(k, (int, float)):
        if k <= 0:
            raise ValueError("The strike price must be a non-negative number.")
    else:
        raise ValueError("The strike price must be a non-negative number.")


def check_r(r):
    if isinstance(r, (int, float)):
        if r < 0:
            raise ValueError(
                "The risk-free rate must be a non-negative number."
            )
    else:
        raise ValueError("The risk-free rate must be a non-negative number.")


def check_expiration(expiration):
    if isinstance(expiration, (int, float)):
        if expiration <= 0:
            raise ValueError(
                "The expiration time must be a non-negative number."
            )
    elif isinstance(expiration, str):
        try:
            expiration = pd.to_datetime(expiration)
        except ValueError:
            raise ValueError(
                "The expiration date must be a valid date string or a non-negative number."
            )
    else:
        raise ValueError(
            "The expiration date must be a valid date string or a non-negative number."
        )

    return expiration


def check_sigma(sigma):
    if isinstance(sigma, (int, float)):
        if sigma <= 0:
            raise ValueError("The volatility must be a non-negative number.")
    elif isinstance(sigma, NoneType):
        pass
    else:
        raise ValueError(
            "The volatility must be None or a non-negative number."
        )


def check_q(q):
    if isinstance(q, (int, float)):
        if q < 0:
            raise ValueError(
                "The dividend yield must be a non-negative number."
            )
    elif q is None:
        pass
    else:
        raise ValueError("The dividend yield must be a non-negative number.")


class PricingModels:

    def __init__(
        self,
        s: str | int | float,
        k: int | float,
        r: float,
        expiration: str | float | int,
        sigma: int | float | None = None,
        q: int | float | None = None,
    ):

        check_types("s", s, [str, float, int])
        check_types("k", k, [int, float])
        check_types("r", r, [int, float])
        check_types("expiration", expiration, [str, float, int])
        check_types("sigma", sigma, [int, float, NoneType])
        check_types("q", q, [int, float, NoneType])

        self._s = check_s(s)

        self.ticker = self._s["ticker"]
        self.ticker_history_1y = self._s["data"]
        if self.ticker:
            try:
                self.ticker_dividend_yield = (
                    self.ticker.info["dividendYield"] if self.ticker else None
                )
            except:
                self.ticker_dividend_yield = None
        else:
            self.ticker_dividend_yield = None

        self.s = self._s["s"]

        self.expiration = check_expiration(expiration)

        if isinstance(self.expiration, (int, float)):
            self.t = self.expiration
        else:
            if self.expiration < pd.to_datetime("now"):
                raise ValueError("The expiration date must be in the future.")
            self.t = (self.expiration - pd.to_datetime("now")).days / 365.0

        check_k(k)
        self.k = k

        check_r(r)
        self.r = r

        check_sigma(sigma)
        if isinstance(sigma, (int, float)):
            self.sigma = sigma
        elif isinstance(sigma, NoneType):
            if self.ticker:
                try:
                    self.sigma = (
                        self.ticker_history_1y["Close"].pct_change().std()
                        * np.sqrt(252)
                        if not self.ticker_history_1y.empty
                        else 0.2
                    )
                except:
                    raise ValueError(
                        "The ticker does not have enough data to calculate the volatility."
                    )

            else:
                raise ValueError(
                    "The volatility must be a non-negative number."
                )

        check_q(q)

        if q is None:
            self.q = (
                float(self.ticker_dividend_yield) / 100
                if self.ticker_dividend_yield
                else 0
            )
        else:
            self.q = q

        if self.q == 0:
            self.model = "black-scholes"
            self.options = models.BlackScholes(
                self.s, self.k, self.r, self.t, self.sigma
            ).__dict__

        else:
            self.model = "black-scholes-merton"
            self.options = models.BlackScholesMerton(
                self.s, self.k, self.r, self.t, self.q, self.sigma
            ).__dict__

        call_keys = [
            key
            for key in self.options.keys()
            if key.endswith("_c") or key == "c"
        ]
        put_keys = [
            key
            for key in self.options.keys()
            if key.endswith("_p") or key == "p"
        ]

        self.call = self.options.copy()
        self.put = self.options.copy()

        for key in call_keys:
            self.put.pop(key)
        for key in put_keys:
            self.call.pop(key)
