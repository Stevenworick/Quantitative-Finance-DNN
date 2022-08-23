import warnings

from curr_pair import CurrPair
from buy_sell import BuySell
from new_cancel import NewCancel
from quote import Quote


class LimitOrderBook:
    # This order book's curr pair.
    __curr_pair: CurrPair
    # Sorted dictionary with prices
    __limit_bids: dict
    __limit_offers: dict
    # All orders
    __all_limit_orders: list
    # Best bid, offer
    __best_bid_price: float
    __best_offer_price: float
    __best_bid: Quote
    __best_offer: Quote
    # Bid/Offer is set (once)
    __is_not_set_bid = True
    __is_not_set_offer = True

    def __init__(self, curr_pair: CurrPair):
        self.__curr_pair = curr_pair
        self.__limit_bids = {}
        self.__limit_offers = {}
        self.__all_limit_orders = []

    def on_new_order(self, quote: Quote):
        if quote.way() == BuySell.BUY:
            self._insert_in_bids(quote)
        else:
            self._insert_in_offers(quote)

    def on_cancel_order(self, quote: Quote):
        self._remove_order(quote.id())

    def count_bids(self):
        bids: int = 0
        for each_order in self.__all_limit_orders:
            if each_order.way() == BuySell.BUY:
                bids += 1
        return bids

    def count_offers(self):
        offers: int = 0
        for each_order in self.__all_limit_orders:
            if each_order.way() == BuySell.SELL:
                offers += 1
        return offers

    def get_best_bid_price(self) -> float:
        if self.__is_not_set_bid:
            return 0.00
        return self.__best_bid_price

    def get_best_bid(self) -> Quote:
        if self.__is_not_set_bid:
            return None
        return self.__best_bid

    def get_best_offer_price(self) -> float:
        if self.__is_not_set_offer:
            return 0.00
        return self.__best_offer_price

    def get_best_offer(self) -> Quote:
        if self.__is_not_set_offer:
            return None
        return self.__best_offer

    def get_best_orders_by_amount(self, way: BuySell, amount: float) -> Quote:
        """
        Returns the best limit order for the given way and amount
        :param way: Buy (for BID side); Sell (for OFFER side)
        :param amount: the amount that has to be bought/sold on market.
        :return: the order that matches the currency pair; amount and side.
        """
        # There are no orders in the order book yet
        if (way == BuySell.BUY and self.__is_not_set_bid) or (way == BuySell.SELL and self.__is_not_set_offer):
            return None
        each_order: Quote
        retained_order: Quote = None
        for each_order in self.__all_limit_orders:
            # Match volume
            if each_order.way() == way and each_order.amount() >= amount:
                # First time check
                if retained_order is None:
                    retained_order = each_order
                # Check price
                elif (way == BuySell.BUY and retained_order.price() < each_order.price()) or \
                     (way == BuySell.SELL and retained_order.price() > each_order.price()):
                    retained_order = each_order
        return retained_order

    def _insert_in_bids(self, quote: Quote):
        """
        Inserts the order in the list of bids
        :param quote: the quote to insert
        :return:
        """
        # Flag once
        if self.__is_not_set_bid:
            self.__best_bid = quote
            self.__best_bid_price = quote.price()
            self.__is_not_set_bid = False

        # Check all the limits: if the ID was inserted previously -> skip
        for each_order in self.__all_limit_orders:
            if each_order.id() == quote.id():
                warnings.warn("Duplicate ID added. Skipping", RuntimeWarning)
                return

        # Check that the price is a new level.
        if quote.price() not in self.__limit_bids.keys():
            # Create a new list. Insert.
            self.__limit_bids[quote.price()] = [quote]
        else:
            # This bid doesn't exist in the collection. Add it to the end.
            self.__limit_bids[quote.price()].append(quote)

        self.__all_limit_orders.append(quote)

        # Update the most used statistics
        if self.__best_bid.compare(quote) > 0:
            self.__best_bid = quote
            self.__best_bid_price = quote.price()

    def _insert_in_offers(self, quote: Quote):
        """
        Inserts the order in the list of bids
        :param quote: the quote to insert
        :return:
        """
        # Flag once
        if self.__is_not_set_offer:
            self.__best_offer = quote
            self.__best_offer_price = quote.price()
            self.__is_not_set_offer = False

        # Check all the limits: if the ID was inserted previously -> skip
        for each_order in self.__all_limit_orders:
            if each_order.id() == quote.id():
                warnings.warn("Duplicate ID added. Skipping", RuntimeWarning)
                return

        # Check that the price is a new level.
        if quote.price() not in self.__limit_offers.keys():
            # Create a new list. Insert.
            self.__limit_offers[quote.price()] = [quote]
        else:
            # This offer doesn't exist in the collection. Add it to the end.
            self.__limit_offers[quote.price()].append(quote)

        self.__all_limit_orders.append(quote)

        # Update the most used statistics
        if self.__best_offer.compare(quote) > 0:
            self.__best_offer = quote
            self.__best_offer_price = quote.price()

    def _remove_order(self, quote_id: int):
        """
        Finds the order by its ID in the whole collection and removes it
        :param quote_id:
        :return:
        """
        order_found: Quote = None
        for order in self.__all_limit_orders:
            if order.id() == quote_id:
                order_found = order
                break

        if order_found is None:
            raise RuntimeError("The order with quote ID {} wasn't found".format(quote_id))

        if order_found.way() == BuySell.BUY:
            self._remove_from_bids(order_found)
        else:
            self._remove_from_offers(order_found)

        self.__all_limit_orders.remove(order_found)

    def _remove_from_bids(self, quote):
        """
        Removes the given order from the bids, updates best order if needed
        :param quote: removed order
        """
        is_updating_best_bid = False
        if self.__best_bid == quote:
            is_updating_best_bid = True

        # Remove the order from the limit
        self.__limit_bids[quote.price()].remove(quote)
        # There are no bids left on this limit => remove it from the dictionary
        if len(self.__limit_bids[quote.price()]) == 0:
            self.__limit_bids.pop(quote.price())

        # Check that there is still some data in the collection.
        if len(self.__limit_bids) == 0:
            self.__is_not_set_offer = True

        # Find new best bid
        if is_updating_best_bid and not self.__is_not_set_offer:
            new_best_bid = 0.00
            for next_bid in self.__limit_bids.keys():
                if new_best_bid < next_bid:
                    new_best_bid = next_bid

            self.__best_bid_price = new_best_bid
            # There's a list behind => take its first element.
            self.__best_bid = self.__limit_bids[new_best_bid][0]

    def _remove_from_offers(self, quote):
        """
        Removes the given order from the offers, updates best order if needed
        :param quote: removed order
        """
        is_updating_best_offer = False
        if self.__best_offer == quote:
            is_updating_best_offer = True

        # Remove the order from the limit
        self.__limit_offers[quote.price()].remove(quote)
        # There are no bids left on this limit => remove it from the dictionary
        if len(self.__limit_offers[quote.price()]) == 0:
            self.__limit_offers.pop(quote.price())

        # Check that there is still some data in the collection.
        if len(self.__limit_offers) == 0:
            self.__is_not_set_offer = True

        # Find new best bid
        if is_updating_best_offer and not self.__is_not_set_offer:
            new_best_offer = list(self.__limit_offers.keys())[0]
            for next_offer in self.__limit_offers.keys():
                if new_best_offer > next_offer:
                    new_best_offer = next_offer

            self.__best_offer_price = new_best_offer
            # There's a list behind => take its first element.
            self.__best_offer = self.__limit_offers[new_best_offer][0]
