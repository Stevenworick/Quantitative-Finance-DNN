from datetime import datetime
from curr_pair import CurrPair, read_string_rep
from buy_sell import BuySell
from new_cancel import NewCancel
from math import fabs, floor, ceil


class Quote:
    # This class encapsulates all the necessary information for a specific quote.
    __quote_id: int
    # Time of the quote (to set in init).
    __quote_time: int
    # Close price.
    __quote_px: float
    # Amount
    __quote_amount: float
    # Ticker currency pair
    __curr_pair: CurrPair
    # Order way
    __order_way: BuySell
    # Order type (CANCEL or NEW)
    __order_type: NewCancel

    def __init__(self, string_representation: str):
        """

        :param string_representation:
            N;101;EUR/USD;39136466000000;1610963536443;100000.00;0.00;0.00;1.20615;B;0
            or:
            C;117;USD/CHF;39136477133462;1610963536459
            goes as:
            ORDERTYPE(N/M/C);ORDER ID;PAIR;LOCAL TIMESTAMP;EXCHANGE TIMESTAMP;AMOUNT;MINQTY;LOTSIZE;PRICE;WAY(B/S);SCOPE

            MINQTY;LOTSIZE;SCOPE ARE IGNORED AND ALWAYS NIL

            N stands for NEW
            C stands for CANCEL

            B stands for Bid
            S stands for Offer
        """
        split_row = string_representation.split(sep=';', maxsplit=10)
        # Save info
        self.__order_type = NewCancel.NEW if split_row[0] == 'N' else NewCancel.CANCEL
        # Common fields for NEW and CANCEL
        self.__quote_id = int(split_row[1])
        self.__curr_pair = read_string_rep(split_row[2])
        self.__quote_time = int(split_row[3])
        # Fields for NEW only
        if self.__order_type == NewCancel.NEW:
            self.__quote_amount = float(split_row[5])
            self.__quote_px = float(split_row[8])
            self.__order_way = BuySell.BUY if split_row[9] == 'B' else BuySell.SELL

    def id(self) -> int:
        """
        Returns the ID of this specific quote
        :return:
        """
        return self.__quote_id

    def price(self) -> float:
        """
        Returns the price corresponding to a specific datetime
        :return:
        """
        return self.__quote_px

    def amount(self) -> float:
        """
        Returns the amount assosiciated to this price.
        :return:
        """
        return self.__quote_amount

    def time(self) -> datetime:
        """
        Returns the datetime of this specifiq quote
        :return:
        """
        return self.__quote_time

    def currency_pair(self) -> int:
        """
        Returns the currency pair of this quote
        :return:
        """
        return self.__curr_pair

    def way(self) -> BuySell:
        """
        Returns whether it's a Buy or a Sell
        :return:
        """
        return self.__order_way

    def type(self) -> NewCancel:
        """
        Return the type of the order (new or cancel)
        :return:
        """
        return self.__order_type

    # COMPARISON METHODS

    def compare(self, other):
        """
        Compares two orders by their price
        :param other: order Quote order
        :return: either -1 +1 or 0
        """
        if self.__order_type != other.type() or self.__order_way != other.way():
            raise RuntimeError("Please compare only NEW orders with the same way (both BUY or SELL).")

        if self.__eq__(other):
            return 0

        if self.__order_way == BuySell.BUY:
            price_diff: float = other.price() - self.__quote_px
        else:
            price_diff: float = self.__quote_px - other.price()

        if fabs(price_diff) <= 0.0000001:
            return 0

        return floor(price_diff * 10000000.00) if price_diff < 0.0 else ceil(price_diff * 10000000.00)

    def __lt__(self, other):
        if self.__eq__(other):
            return True

        if self.__order_way == BuySell.BUY:
            price_diff: float = other.price() - self.__quote_px
        else:
            price_diff: float = self.__quote_px - other.price()

        if fabs(price_diff) <= 0.0000001:
            return True

        return price_diff < 0

    def __cmp__(self, other):
        return self.compare(other)

    def __eq__(self, other):
        return self.__quote_id == other.id() and self.__order_type == other.type()
