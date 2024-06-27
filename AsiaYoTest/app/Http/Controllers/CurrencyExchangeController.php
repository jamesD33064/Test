<?php

namespace App\Http\Controllers;

use App\Services\CurrencyExchangeService;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Validator;

class CurrencyExchangeController extends Controller
{
    private $currencyExchangeService;

    public function __construct(CurrencyExchangeService $currencyExchangeService)
    {
        $this->currencyExchangeService = $currencyExchangeService;
    }

    public function convert(Request $request)
    {
        $validator = Validator::make($request->all(), [
            'source' => 'required|string',
            'target' => 'required|string',
            'amount' => 'required|regex:#^[0-9,.]+$#',
        ]);

        if ($validator->fails()) {
            return response()->json([
                'msg' => $validator->errors(),
            ], 400);
        }

        $source = $request->input('source');
        $target = $request->input('target');
        $amount = $request->input('amount');

        try {
            $convertedAmount = $this->currencyExchangeService->convert($source, $target, $amount);
            $formattedAmount = number_format($convertedAmount, 2);

            return response()->json([
                'msg' => 'success',
                'amount' => $formattedAmount,
            ]);
        } catch (\InvalidArgumentException $e) {
            return response()->json([
                'msg' => $e->getMessage(),
            ], 400);
        }
    }
}