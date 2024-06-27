<?php

namespace Tests\Unit;

use App\Services\CurrencyExchangeService;
use PHPUnit\Framework\TestCase;

class CurrencyExchangeServiceTest extends TestCase
{
    private $currencyExchangeService;

    protected function setUp(): void
    {
        parent::setUp();

        $this->currencyExchangeService = new CurrencyExchangeService([
            'TWD' => ['TWD' => 1, 'JPY' => 3.669, 'USD' => 0.03281],
            'JPY' => ['TWD' => 0.26956, 'JPY' => 1, 'USD' => 0.00885],
            'USD' => ['TWD' => 30.444, 'JPY' => 111.801, 'USD' => 1],
        ]);
    }

    public function testConvertWithUnsupportedCurrency()
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Unsupported currency');

        $this->currencyExchangeService->convert('EUR', 'USD', '100');
    }

    public function testConvertWithInvalidAmount()
    {
        $this->expectException(\InvalidArgumentException::class);
        $this->expectExceptionMessage('Invalid amount format');

        $this->currencyExchangeService->convert('USD', 'JPY', 'abc');
    }

    public function testConvertWithIntegerUsd2Twd()
    {
        $result = $this->currencyExchangeService->convert('USD', 'TWD', '100');
        $this->assertEquals(3044.40, $result);
    }
    
    public function testConvertWithDecimalUsd2Twd()
    {
        $result = $this->currencyExchangeService->convert('USD', 'TWD', '100.50');
        $this->assertEquals(3059.62, $result);
    }
    
    public function testConvertWithCommaUsd2Twd()
    {
        $result = $this->currencyExchangeService->convert('USD', 'TWD', '1,000');
        $this->assertEquals(30444.00, $result);
    }
    
    public function testConvertWithIntegerJpy2Usd()
    {
        $result = $this->currencyExchangeService->convert('JPY', 'USD', '1000');
        $this->assertEquals(8.85, $result);
    }
    
    public function testConvertWithDecimalJpy2Usd()
    {
        $result = $this->currencyExchangeService->convert('JPY', 'USD', '1000.50');
        $this->assertEquals(8.85, $result);
    }
    
    public function testConvertWithCommaJpy2Usd()
    {
        $result = $this->currencyExchangeService->convert('JPY', 'USD', '10,000');
        $this->assertEquals(88.5, $result);
    }
    
    public function testConvertWithIntegerTwd2Jpy()
    {
        $result = $this->currencyExchangeService->convert('TWD', 'JPY', '1000');
        $this->assertEquals(3669.00, $result);
    }
    
    public function testConvertWithDecimalTwd2Jpy()
    {
        $result = $this->currencyExchangeService->convert('TWD', 'JPY', '1000.50');
        $this->assertEquals(3670.83, $result);
    }
    
    public function testConvertWithCommaTwd2Jpy()
    {
        $result = $this->currencyExchangeService->convert('TWD', 'JPY', '10,000');
        $this->assertEquals(36690.00, $result);
    }
}