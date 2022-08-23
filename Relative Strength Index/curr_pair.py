# Enumerates currency pairs
class CurrPair:
    EURUSD = 1
    GBPUSD = 2
    USDCHF = 3
    USDJPY = 4
    EURJPY = 5
    AUDUSD = 6
    USDJPY = 7
    NOKSEK = 8
    USDCAD = 9


# Holds string representation for all CCY Pairs. Used as an alternative to the C SWITCH
dict_all_values = {
    "EUR/USD": CurrPair.EURUSD,
    "GBP/USD": CurrPair.GBPUSD,
    "USD/CHF": CurrPair.USDCHF,
    "USD/JPY": CurrPair.USDJPY,
    "EUR/JPY": CurrPair.EURJPY,
    "AUD/USD": CurrPair.AUDUSD,
    "USD/JPY": CurrPair.USDJPY,
    "NOK/SEK": CurrPair.NOKSEK,
    "USD/CAD": CurrPair.USDCAD
}


def read_string_rep(string_rep: str) -> CurrPair:
    """
    Reads the string representation in form of XXX/YYY and returns an ENUM with the correct representation in numerical
    form CurrPair.XXXYYY
    :param string_rep: XXX/YYY
    :return:
    """
    if string_rep in dict_all_values:
        return dict_all_values[string_rep]
    else:
        raise RuntimeError("Please add {} to the CurrPair enum class and dict_all_values collection.".format(string_rep))
