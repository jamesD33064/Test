<?php

namespace App\Services;

class CurrencyExchangeService
{
    private $rates;

    public function __construct(array $rates)
    {
        $this->rates = $rates;
    }

    public function convert(string $source, string $target, string $amount): float
    {
        // 移除千分位記號
        $amount = str_replace(',', '', $amount);

        // 驗證金額是否為數字
        if (!is_numeric($amount)) {
            throw new \InvalidArgumentException('Invalid amount format');
        }
        // 驗證貨幣代碼是否有在rates中
        if (!isset($this->rates[$source]) || !isset($this->rates[$target])) {
            throw new \InvalidArgumentException('Unsupported currency');
        }

        $sourceRate = $this->rates[$source];

        $convertedAmount = ($amount * $sourceRate[$target]);

        return round($convertedAmount, 2);
    }
}