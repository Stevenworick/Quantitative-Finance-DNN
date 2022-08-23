import warnings

from quote import Quote
from limit_order_book import LimitOrderBook
from buy_sell import BuySell


class TradeSituation:
    # This variable will be incremented after each call of TradeSituation.generate_next_id().
    # It is used to populate __trade_situation_id.
    __common_trade_situation_id: int = 0
    # This is a global reference to the order book
    __common_order_book: LimitOrderBook
    # Instance attributes
    # Unique ID of the trade_situation
    __trade_situation_id: int
    # If True: it's a LONG (BUY) trade. If False: it's a SHORT (SELL) trade.
    __is_long_trade: bool
    # Quote saved when we opened the position
    __executed_open_quote: Quote
    # Reference quote when position opened.
    __arrived_open_quote: Quote
    # Quote saved when we close the position
    __executed_close_quote: Quote
    # Reference quote when position closed.
    __arrived_close_quote: Quote
    # Flag used to describe if the position is opened or closed
    __is_closed: bool
    # Maximum draw down in basis points. Always positive!
    __max_dd_in_bps: float
    # Latest profit or loss of the position in basis points
    __pnl_bps: float
    # Take profit in basis points
    __take_profit_in_bps: float
    # Trading amount
    __amount: float
    # This variable describes that we are using the best BID and best OFFER to calculate PnL
    __is_best_price_calculation: bool

    def __init__(self, open_order_arg: Quote, is_long_trade_arg: bool, take_profit_in_bps_arg: float, amount: float,
                 is_best_px_calc: bool):
        # Init locals
        self.__max_dd_in_bps = 0.00
        self.__pnl_bps = 0.00
        self.__is_closed = True
        # Update and set the __trade_situation_id
        self.__trade_situation_id = TradeSituation.generate_next_id()
        # Check arguments sanity.
        if take_profit_in_bps_arg < 0.00:
            raise Exception("Please note that the take profit has to be positive (:2.2f)"
                            .format(take_profit_in_bps_arg))
        # Set up the rest of variables
        self.__is_best_price_calculation = is_best_px_calc
        self.__is_long_trade = is_long_trade_arg
        self.__take_profit_in_bps = take_profit_in_bps_arg
        self.__amount = amount
        # Call self.open_position(...) to open the position immediately
        self.open_position(open_order_arg)

    def open_position(self, quote_arg: Quote):
        """
        Flags the is_closed to False. Saves the entry order.
        :param quote_arg: quote class's instance expected. The first quote.
        :return:
        """
        # Sets the __executed_open_quote to argument's value and flags __is_closed to FALSE
        opening_quote_way: BuySell = BuySell.SELL if self.__is_long_trade else BuySell.BUY
        self.__executed_open_quote = TradeSituation.__common_order_book.get_best_orders_by_amount(opening_quote_way,
                                                                                                  self.__amount)
        self.__arrived_open_quote = quote_arg
        self.__is_closed = False

    def close_position(self, quote_arg: Quote):
        """
        Flags the position as closed. Calculates final PnL
        :param quote_arg: last quote
        :return:
        """
        # Reference quote
        self.__arrived_close_quote = quote_arg
        # Sets the __executed_close_quote to argument's value, flags __is_closed to TRUE
        if self.__is_long_trade:
            self.__executed_close_quote = TradeSituation.__common_order_book.get_best_orders_by_amount(BuySell.BUY,
                                                                                                       self.__amount)
        else:
            self.__executed_close_quote = TradeSituation.__common_order_book.get_best_orders_by_amount(BuySell.SELL,
                                                                                                       self.__amount)
        if self.__executed_close_quote is not None:
            if self.__is_long_trade:
                # Buy with Offer, close the position with Bid
                self.__pnl_bps = self.__executed_close_quote.price() - self.__executed_open_quote.price()
            else:
                # Sell with Bid, close the position with Offer
                self.__pnl_bps = self.__executed_open_quote.price() - self.__executed_close_quote.price()
            # otherwise keep the approx PNL
        else:
            warnings.warn("Could not retrieve the corresponding order to close the position", RuntimeWarning)

        self.__is_closed = True

    def update_on_order(self, quote_arg: Quote) -> bool:
        """
        Updates all the variables in the position. Calculates the PnL.
        :param quote_arg: the latest quote
        :return: returns True if the position was closed (target profit reached)
        """
        # Check if the position is alive. Return false if the position is dormant
        if self.__is_closed:
            return False
        # Check/update current pnl and draw down
        self.calculate_pnl_and_dd()
        # Check if target pnl was reached
        if self.__pnl_bps >= self.__take_profit_in_bps:
            # Target pnl reached: close position
            self.close_position(quote_arg)
            return True

        # PnL target not reached: return false
        return False

    def calculate_pnl_and_dd(self) -> float:
        """
        Calculates (and updates) the PnL and draw down for the position
        :param quote_arg: the current quote. Given only for statistics purpose. The best price is kept in the order book
        :return: current pnl
        """
        # In case the position is not opened (not alive) return the value stored in __pnl_bps
        if self.__is_closed:
            return self.__pnl_bps

        # Calculate pnl (different for LONG and SHORT; if we use the best price or not)
        if self.__is_best_price_calculation:
            # Get the best price on market (faster)
            if self.__is_long_trade:
                if TradeSituation.__common_order_book.get_best_bid() is not None:
                    price_reference = TradeSituation.__common_order_book.get_best_bid_price()
                else:
                    # No price available
                    return self.__pnl_bps
            else:
                if TradeSituation.__common_order_book.get_best_offer() is not None:
                    price_reference = TradeSituation.__common_order_book.get_best_offer_price()
                else:
                    # No price available
                    return self.__pnl_bps
        else:
            # Get the price by amount (slower)
            if self.__is_long_trade:
                corresponding_order = TradeSituation.__common_order_book.get_best_orders_by_amount(BuySell.BUY,
                                                                                                   self.__amount)
            else:
                corresponding_order = TradeSituation.__common_order_book.get_best_orders_by_amount(BuySell.SELL,
                                                                                                   self.__amount)
            if corresponding_order is not None:
                price_reference = corresponding_order.price()
            else:
                # No price available
                return self.__pnl_bps

        if self.__is_long_trade:
            # Buy with Offer, close the position with Bid
            self.__pnl_bps = price_reference - self.__executed_open_quote.price()
        else:
            # Sell with Bid, close the position with Offer
            self.__pnl_bps = self.__executed_open_quote.price() - price_reference

        # Calculate draw down
        if self.__pnl_bps < 0.00 and -self.__pnl_bps > self.__max_dd_in_bps:
            self.__max_dd_in_bps = -self.__pnl_bps

        # return __pnl_bps
        return self.__pnl_bps

    def return_current_pnl(self) -> float:
        """
        Returns the current (or final if the position is closed) pnl.
        :return:
        """
        return self.__pnl_bps

    def return_current_draw_down(self) -> float:
        """
        Returns the current (or final if the position is closed) maximum draw down.
        :return:
        """
        return self.__max_dd_in_bps

    def trade_situation_id(self) -> int:
        """
        Returns this trade situation ID
        :return:
        """
        return self.__trade_situation_id

    def is_closed(self):
        """
        Returns true if the position was closed previously
        :return:
        """
        return self.__is_closed

    @staticmethod
    def generate_next_id():
        TradeSituation.__common_trade_situation_id += 1
        return TradeSituation.__common_trade_situation_id

    @staticmethod
    def set_limit_order_book(limit_order_book: LimitOrderBook):
        TradeSituation.__common_order_book = limit_order_book
