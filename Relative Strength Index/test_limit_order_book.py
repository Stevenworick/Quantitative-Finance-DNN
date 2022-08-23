from unittest import TestCase
from quote import Quote
from curr_pair import CurrPair
from limit_order_book import LimitOrderBook
from buy_sell import BuySell


class TestLimitOrderBook(TestCase):

    def test_insert_cancel_sort(self):
        quote1_b = Quote("N;112;USD/CHF;39136476157873;1610963536459;100000.00;0.00;0.00;0.89153;B;0")
        quote2_b = Quote("N;117;USD/CHF;39136474132701;1610963536445;1000000.00;0.00;0.00;0.89154;B;0")
        quote3_b = Quote("N;118;USD/CHF;39136474135095;1610963536445;2000000.00;0.00;0.00;0.89154;B;0")
        quote4_b = Quote("N;119;USD/CHF;39136474135097;1610963536445;2000000.00;0.00;0.00;0.89152;B;0")

        quote1_s = Quote("N;212;USD/CHF;39136476157874;1610963536459;100000.00;0.00;0.00;0.89173;S;0")
        quote2_s = Quote("N;219;USD/CHF;39136474135098;1610963536445;2000000.00;0.00;0.00;0.89176;S;0")
        quote3_s = Quote("N;220;USD/CHF;39136474268689;1610963536445;3000000.00;0.00;0.00;0.89179;S;0")
        quote4_s = Quote("N;221;USD/CHF;39136474268691;1610963536445;5000000.00;0.00;0.00;0.89183;S;0")

        order_book = LimitOrderBook(CurrPair.USDCHF)
        order_book.on_new_order(quote1_b)
        order_book.on_new_order(quote2_b)
        order_book.on_new_order(quote3_b)
        order_book.on_new_order(quote4_b)

        order_book.on_new_order(quote1_s)
        order_book.on_new_order(quote2_s)
        order_book.on_new_order(quote3_s)
        order_book.on_new_order(quote4_s)

        self.assertEqual(0.89154, order_book.get_best_bid_price())
        self.assertEqual(0.89173, order_book.get_best_offer_price())

        self.assertGreater(order_book.count_bids(), 3)
        self.assertGreater(order_book.count_offers(), 3)

        self.assertEqual(118, order_book.get_best_orders_by_amount(BuySell.BUY, 2000000.00).id())
        self.assertEqual(117, order_book.get_best_orders_by_amount(BuySell.BUY, 1000000.00).id())

        self.assertEqual(220, order_book.get_best_orders_by_amount(BuySell.SELL, 3000000.00).id())
        self.assertEqual(221, order_book.get_best_orders_by_amount(BuySell.SELL, 5000000.00).id())

        quote3_bc = Quote("C;118;USD/CHF;39136477133464;1610963536459")
        order_book.on_cancel_order(quote3_bc)

        self.assertEqual(119, order_book.get_best_orders_by_amount(BuySell.BUY, 2000000.00).id())

        quote3_sc = Quote("C;220;USD/CHF;39136477259707;1610963536459")
        order_book.on_cancel_order(quote3_sc)

        self.assertGreater(order_book.count_bids(), 2)
        self.assertGreater(order_book.count_offers(), 2)

        quote1_sc = Quote("C;212;USD/CHF;39136477259707;1610963536459")
        order_book.on_cancel_order(quote1_sc)

        self.assertEqual(0.89176, order_book.get_best_offer_price())
        self.assertEqual(219, order_book.get_best_offer().id())



