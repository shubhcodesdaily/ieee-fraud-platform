SELECT
    transactionid,
    card1,
    transactiondt,
    transactionamt,
    AVG(transactionamt) OVER (
        PARTITION BY card1
        ORDER BY transactiondt
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    ) AS card_avg_amt_before_this
FROM transactions
ORDER BY card1, transactiondt
LIMIT 20;
