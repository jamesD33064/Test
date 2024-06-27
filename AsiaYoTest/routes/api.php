<?php

use App\Http\Controllers\CurrencyExchangeController;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

Route::post('/convert', [CurrencyExchangeController::class, 'convert']);
