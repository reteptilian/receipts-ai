from __future__ import annotations

import csv
import json
import sys
from datetime import date
from io import StringIO
from pathlib import Path

import pytest

from receipts_ai import ingest_statements
from receipts_ai.ingest_statements import (
    CSV_FIELDNAMES,
    main,
    transactions_from_fidelity_csv,
    transactions_from_file,
    transactions_from_ofx,
    write_transaction_json,
    write_transactions_csv,
    write_transactions_json,
)
from receipts_ai.models.transaction import (
    CategoryAllocation,
    Kind,
    Source,
    Source1,
    Status,
    Transaction,
)


def test_parses_sgml_ofx_bank_transactions():
    transactions = transactions_from_ofx(
        """
        OFXHEADER:100
        DATA:OFXSGML
        <OFX>
          <BANKMSGSRSV1>
            <STMTTRNRS>
              <STMTRS>
                <CURDEF>USD
                <BANKACCTFROM>
                  <BANKID>123456789
                  <ACCTID>987654321
                  <ACCTTYPE>CHECKING
                </BANKACCTFROM>
                <BANKTRANLIST>
                  <STMTTRN>
                    <TRNTYPE>DEBIT
                    <DTPOSTED>20260427000000[-8:PST]
                    <TRNAMT>-42.19
                    <FITID>2026042701
                    <NAME>COSTCO WHSE
                    <MEMO>POS PURCHASE COSTCO WHSE #123
                  </STMTTRN>
                </BANKTRANLIST>
              </STMTRS>
            </STMTTRNRS>
          </BANKMSGSRSV1>
        </OFX>
        """
    )

    assert len(transactions) == 1
    transaction = transactions[0]
    assert transaction.source == Source.bank_statement
    assert transaction.external_id == "2026042701"
    assert transaction.account_id == "123456789:987654321:CHECKING"
    assert transaction.transaction_date == date(2026, 4, 27)
    assert transaction.payee is None
    assert transaction.description == "POS PURCHASE COSTCO WHSE #123"
    assert transaction.amount == "-42.19"
    assert transaction.currency == "USD"
    assert transaction.kind == Kind.expense
    assert transaction.status == Status.posted
    assert transaction.receipt is None


def test_parses_xml_ofx_credit_card_transactions():
    transactions = transactions_from_ofx(
        """
        <OFX>
          <CREDITCARDMSGSRSV1>
            <CCSTMTTRNRS>
              <CCSTMTRS>
                <CURDEF>usd</CURDEF>
                <CCACCTFROM>
                  <ACCTID>4111111111111111</ACCTID>
                </CCACCTFROM>
                <BANKTRANLIST>
                  <STMTTRN>
                    <TRNTYPE>CREDIT</TRNTYPE>
                    <DTPOSTED>20260428</DTPOSTED>
                    <DTAVAIL>20260429</DTAVAIL>
                    <TRNAMT>1250.00</TRNAMT>
                    <FITID>deposit-1</FITID>
                    <NAME>Payroll</NAME>
                  </STMTTRN>
                </BANKTRANLIST>
              </CCSTMTRS>
            </CCSTMTTRNRS>
          </CREDITCARDMSGSRSV1>
        </OFX>
        """
    )

    assert len(transactions) == 1
    transaction = transactions[0]
    assert transaction.account_id == "4111111111111111"
    assert transaction.currency == "USD"
    assert transaction.posted_date == date(2026, 4, 29)
    assert transaction.payee == "Payroll"
    assert transaction.description is None
    assert transaction.amount == "1250.00"
    assert transaction.kind == Kind.income


def test_parses_sgml_qfx_credit_card_transactions(tmp_path: Path):
    statement_path = tmp_path / "credit-card.qfx"
    statement_path.write_text(
        """
        OFXHEADER:100
        DATA:OFXSGML
        VERSION:102
        SECURITY:NONE
        ENCODING:USASCII
        CHARSET:1252
        COMPRESSION:NONE
        OLDFILEUID:NONE
        NEWFILEUID:statement-1

        <OFX>
          <SIGNONMSGSRSV1>
            <SONRS>
              <STATUS>
                <CODE>0
                <SEVERITY>INFO
              </STATUS>
              <DTSERVER>20260501120000[-8:PST]
              <LANGUAGE>ENG
              <INTU.BID>12345
            </SONRS>
          </SIGNONMSGSRSV1>
          <CREDITCARDMSGSRSV1>
            <CCSTMTTRNRS>
              <TRNUID>1
              <STATUS>
                <CODE>0
                <SEVERITY>INFO
              </STATUS>
              <CCSTMTRS>
                <CURDEF>USD
                <CCACCTFROM>
                  <ACCTID>5555444433331111
                </CCACCTFROM>
                <BANKTRANLIST>
                  <DTSTART>20260401000000
                  <DTEND>20260430000000
                  <STMTTRN>
                    <TRNTYPE>DEBIT
                    <DTPOSTED>20260427120000[-8:PST]
                    <TRNAMT>-42.19
                    <FITID>2026042701
                    <NAME>COSTCO WHSE #123
                    <MEMO>POS PURCHASE COSTCO WHSE #123
                  </STMTTRN>
                  <STMTTRN>
                    <TRNTYPE>CREDIT
                    <DTPOSTED>20260428000000[-8:PST]
                    <TRNAMT>42.19
                    <FITID>2026042801
                    <NAME>ONLINE PAYMENT
                  </STMTTRN>
                </BANKTRANLIST>
              </CCSTMTRS>
            </CCSTMTTRNRS>
          </CREDITCARDMSGSRSV1>
        </OFX>
        """,
        encoding="utf-8",
    )

    transactions = transactions_from_file(statement_path, statement_format="qfx")

    assert len(transactions) == 2
    charge = transactions[0]
    assert charge.source == Source.bank_statement
    assert charge.external_id == "2026042701"
    assert charge.account_id == "5555444433331111"
    assert charge.transaction_date == date(2026, 4, 27)
    assert charge.payee == "COSTCO WHSE #123"
    assert charge.description == "POS PURCHASE COSTCO WHSE #123"
    assert charge.amount == "-42.19"
    assert charge.currency == "USD"
    assert charge.kind == Kind.expense
    assert charge.status == Status.posted

    payment = transactions[1]
    assert payment.payee == "ONLINE PAYMENT"
    assert payment.description is None
    assert payment.amount == "42.19"
    assert payment.kind == Kind.income


def test_extracts_mcc_from_credit_card_memo():
    transactions = transactions_from_ofx(
        """
        <OFX>
          <CREDITCARDMSGSRSV1>
            <CCSTMTTRNRS>
              <CCSTMTRS>
                <CURDEF>USD
                <CCACCTFROM>
                  <ACCTID>5555444433331111
                </CCACCTFROM>
                <BANKTRANLIST>
                  <STMTTRN>
                    <TRNTYPE>DEBIT
                    <DTPOSTED>20260406120000.000
                    <TRNAMT>-649.35
                    <FITID>647d8b3c-cbe0-1dc0-dcc1-8edd029e2a29
                    <NAME>TM *SEATTLE MARINERS-S 800-653-8
                    <MEMO>24692166095105027421606; 07922; ; 0829PETERESBENSEN; ; ;
                  </STMTTRN>
                </BANKTRANLIST>
              </CCSTMTRS>
            </CCSTMTTRNRS>
          </CREDITCARDMSGSRSV1>
        </OFX>
        """
    )

    transaction = transactions[0]
    assert transaction.mcc == "7922"
    assert (
        transaction.mcc_description
        == "Theatrical Producers (Except Motion Pictures), Ticket Agencies"
    )


def test_does_not_extract_mcc_from_bank_account_memo():
    transactions = transactions_from_ofx(
        """
        <OFX>
          <BANKMSGSRSV1>
            <STMTTRNRS>
              <STMTRS>
                <CURDEF>USD
                <BANKACCTFROM>
                  <ACCTID>987654321
                </BANKACCTFROM>
                <BANKTRANLIST>
                  <STMTTRN>
                    <TRNTYPE>DEBIT
                    <DTPOSTED>20260406120000.000
                    <TRNAMT>-649.35
                    <FITID>checking-1
                    <NAME>Memo Pattern
                    <MEMO>24692166095105027421606; 07922; ; ;
                  </STMTTRN>
                </BANKTRANLIST>
              </STMTRS>
            </STMTTRNRS>
          </BANKMSGSRSV1>
        </OFX>
        """
    )

    transaction = transactions[0]
    assert transaction.mcc is None
    assert transaction.mcc_description is None


def test_parses_fidelity_csv_transactions_ignoring_padding_and_boilerplate():
    transactions = transactions_from_fidelity_csv(
        """


Run Date,Action,Symbol,Description,Type,Price ($),Quantity,Commission ($),Fees ($),Accrued Interest ($),Amount ($),Cash Balance ($),Settlement Date
05/01/2026,"DIRECT DEBIT JPMORGAN CHASECHASE ACH (Cash)", ,"No Description",Cash,,0.000,,,,-3562.71,73991.72,
04/30/2026,"REINVESTMENT FIDELITY GOVERNMENT MONEY MARKET (SPAXX) (Cash)",SPAXX,"FIDELITY GOVERNMENT MONEY MARKET",Cash,1,236.53,,,,-236.53,77554.43,
04/30/2026,"DIVIDEND RECEIVED FIDELITY GOVERNMENT MONEY MARKET (SPAXX) (Cash)",SPAXX,"FIDELITY GOVERNMENT MONEY MARKET",Cash,,0.000,,,,236.53,77554.43,


"The data and information in this spreadsheet is provided to you solely for your use and is not for distribution. The spreadsheet is provided for"
"informational purposes only, and is not intended to provide advice"

Date downloaded 05/02/2026 11:08 am
        """
    )

    assert len(transactions) == 3
    transaction = transactions[0]
    assert transaction.source == Source.bank_statement
    assert transaction.external_id is not None
    assert transaction.external_id.startswith("fidelity_csv_")
    assert transaction.account_id is None
    assert transaction.transaction_date == date(2026, 5, 1)
    assert transaction.posted_date is None
    assert transaction.payee is None
    assert transaction.description == "DIRECT DEBIT JPMORGAN CHASECHASE ACH (Cash)"
    assert transaction.amount == "-3562.71"
    assert transaction.currency == "USD"
    assert transaction.kind == Kind.expense
    assert transaction.status == Status.posted
    assert transactions[1].description == (
        "REINVESTMENT FIDELITY GOVERNMENT MONEY MARKET (SPAXX) (Cash) "
        "FIDELITY GOVERNMENT MONEY MARKET"
    )
    assert transactions[2].kind == Kind.income


def test_writes_transaction_csv_rows():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        external_id="fitid-1",
        account_id="acct-1",
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        description="CARD PURCHASE",
        brave_search_result='[{"title":"Coffee Shop","description":"Cafe"}]',
        amount="-7.00",
        currency="USD",
        kind=Kind.expense,
        status=Status.posted,
    )
    output = StringIO()

    write_transactions_csv([transaction], output)

    header, *_ = output.getvalue().splitlines()
    assert header == ",".join(CSV_FIELDNAMES)
    rows = list(csv.DictReader(StringIO(output.getvalue())))
    assert rows == [
        {
            "transaction_id": "bank_statement_1",
            "source": "bank_statement",
            "external_id": "fitid-1",
            "account_id": "acct-1",
            "transaction_date": "2026-04-27",
            "posted_date": "",
            "payee": "Coffee Shop",
            "description": "CARD PURCHASE",
            "mcc": "",
            "mcc_description": "",
            "brave_search_result": '[{"title":"Coffee Shop","description":"Cafe"}]',
            "amount": "-7.00",
            "currency": "USD",
            "kind": "expense",
            "status": "posted",
            "linked_transaction_ids": "[]",
            "category_allocations": "[]",
            "notes": "",
            "created_at": "",
            "updated_at": "",
        }
    ]


def test_transaction_json_output_uses_aliases():
    transaction = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        external_id="fitid-1",
        account_id="acct-1",
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
    )
    output = StringIO()

    write_transaction_json(transaction, output)

    payload = json.loads(output.getvalue())
    assert payload["externalId"] == "fitid-1"
    assert payload["accountId"] == "acct-1"
    assert payload["transactionDate"] == "2026-04-27"
    assert "braveSearchResult" not in payload
    assert "receipt" not in payload


def test_transactions_json_output_writes_json_array():
    transaction_1 = Transaction(
        id="bank_statement_1",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 27),
        payee="Coffee Shop",
        amount="-7.00",
        currency="USD",
    )
    transaction_2 = Transaction(
        id="bank_statement_2",
        source=Source.bank_statement,
        transaction_date=date(2026, 4, 28),
        payee="Payroll",
        amount="1250.00",
        currency="USD",
    )
    output = StringIO()

    write_transactions_json([transaction_1, transaction_2], output)

    payload = json.loads(output.getvalue())
    assert [transaction["id"] for transaction in payload] == [
        "bank_statement_1",
        "bank_statement_2",
    ]


def test_main_processes_multiple_statement_files_as_combined_csv(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    statement_1_path = tmp_path / "checking.ofx"
    statement_2_path = tmp_path / "credit.ofx"
    statement_1_path.write_text(
        """
        <OFX><BANKMSGSRSV1><STMTTRNRS><STMTRS><CURDEF>USD
        <BANKACCTFROM><ACCTID>checking</ACCTID></BANKACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20260427
        <TRNAMT>-7.00<FITID>checking-1<NAME>Coffee Shop</STMTTRN></BANKTRANLIST>
        </STMTRS></STMTTRNRS></BANKMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )
    statement_2_path.write_text(
        """
        <OFX><CREDITCARDMSGSRSV1><CCSTMTTRNRS><CCSTMTRS><CURDEF>USD
        <CCACCTFROM><ACCTID>credit</ACCTID></CCACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>CREDIT<DTPOSTED>20260428
        <TRNAMT>1250.00<FITID>credit-1<NAME>Payroll</STMTTRN></BANKTRANLIST>
        </CCSTMTRS></CCSTMTTRNRS></CREDITCARDMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        ["ingest_statements.py", str(statement_1_path), str(statement_2_path)],
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert [row["payee"] for row in rows] == ["", "Payroll"]
    assert [row["description"] for row in rows] == ["Coffee Shop", ""]
    assert [row["amount"] for row in rows] == ["-7.00", "1250.00"]


def test_main_can_process_fidelity_csv_statement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    statement_path = tmp_path / "fidelity.csv"
    statement_path.write_text(
        """

Run Date,Action,Symbol,Description,Type,Price ($),Quantity,Commission ($),Fees ($),Accrued Interest ($),Amount ($),Cash Balance ($),Settlement Date
05/01/2026,"DIRECT DEBIT JPMORGAN CHASECHASE ACH (Cash)", ,"No Description",Cash,,0.000,,,,-3562.71,73991.72,
Date downloaded 05/02/2026 11:08 am
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_statements.py",
            "--statement-format",
            "fidelity-csv",
            str(statement_path),
        ],
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert [row["transaction_date"] for row in rows] == ["2026-05-01"]
    assert [row["description"] for row in rows] == [
        "DIRECT DEBIT JPMORGAN CHASECHASE ACH (Cash)"
    ]
    assert [row["amount"] for row in rows] == ["-3562.71"]


def test_main_can_process_qfx_statement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    statement_path = tmp_path / "credit-card.qfx"
    statement_path.write_text(
        """
        OFXHEADER:100
        DATA:OFXSGML
        <OFX><CREDITCARDMSGSRSV1><CCSTMTTRNRS><CCSTMTRS><CURDEF>USD
        <CCACCTFROM><ACCTID>credit</ACCTID></CCACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20260427
        <TRNAMT>-7.00<FITID>credit-1<NAME>Coffee Shop<MEMO>CARD PURCHASE</STMTTRN></BANKTRANLIST>
        </CCSTMTRS></CCSTMTTRNRS></CREDITCARDMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_statements.py",
            "--statement-format",
            "qfx",
            str(statement_path),
        ],
    )

    main()

    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    assert rows[0]["account_id"] == "credit"
    assert rows[0]["payee"] == "Coffee Shop"
    assert rows[0]["description"] == "CARD PURCHASE"
    assert rows[0]["amount"] == "-7.00"


def test_main_can_upsert_firestore(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    statement_path = tmp_path / "checking.ofx"
    statement_path.write_text(
        """
        <OFX><BANKMSGSRSV1><STMTTRNRS><STMTRS><CURDEF>USD
        <BANKACCTFROM><ACCTID>checking</ACCTID></BANKACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20260427
        <TRNAMT>-7.00<FITID>checking-1<NAME>Coffee Shop</STMTTRN></BANKTRANLIST>
        </STMTRS></STMTTRNRS></BANKMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )
    calls: list[tuple[str | None, str]] = []

    def fake_upsert_transaction_to_firestore(
        transaction: Transaction, *, collection: str
    ) -> None:
        calls.append((transaction.payee, collection))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_statements.py",
            "--upsert-firestore",
            "--firestore-collection",
            "test-transactions",
            str(statement_path),
        ],
    )
    monkeypatch.setattr(
        ingest_statements,
        "upsert_transaction_to_firestore",
        fake_upsert_transaction_to_firestore,
    )

    main()

    assert calls == [(None, "test-transactions")]


def test_main_can_enrich_and_categorize_transactions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    statement_path = tmp_path / "checking.ofx"
    statement_path.write_text(
        """
        <OFX><BANKMSGSRSV1><STMTTRNRS><STMTRS><CURDEF>USD
        <BANKACCTFROM><ACCTID>checking</ACCTID></BANKACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20260427
        <TRNAMT>-7.00<FITID>checking-1<NAME>Coffee Shop<MEMO>CARD PURCHASE COF SHOP</STMTTRN></BANKTRANLIST>
        </STMTRS></STMTTRNRS></BANKMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_enrich_transactions_with_brave_search(
        transactions: list[Transaction],
        *,
        request_delay_seconds: float | None = None,
    ) -> list[Transaction]:
        calls.append(f"brave:{request_delay_seconds}")
        transactions[0].brave_search_result = json.dumps(
            [{"title": "Coffee Shop", "description": "Cafe and espresso bar."}]
        )
        return transactions

    def fake_categorize_transactions(transactions: list[Transaction]) -> list[Transaction]:
        calls.append("categorize")
        assert transactions[0].brave_search_result is not None
        transactions[0].category_allocations = [
            CategoryAllocation(
                category_id="Food & Dining > Restaurants & Dining Out",
                amount="-7.00",
                confidence=0.8,
                source=Source1.model,
            )
        ]
        return transactions

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_statements.py",
            "--categorize",
            "--brave-search-delay-seconds",
            "1.1",
            str(statement_path),
        ],
    )
    monkeypatch.setattr(
        ingest_statements,
        "enrich_transactions_with_brave_search",
        fake_enrich_transactions_with_brave_search,
    )
    monkeypatch.setattr(ingest_statements, "categorize_transactions", fake_categorize_transactions)

    main()

    assert calls == ["brave:1.1", "categorize"]
    rows = list(csv.DictReader(StringIO(capsys.readouterr().out)))
    allocations = json.loads(rows[0]["category_allocations"])
    assert allocations == [
        {
            "categoryId": "Food & Dining > Restaurants & Dining Out",
            "amount": "-7.00",
            "confidence": 0.8,
            "source": "model",
        }
    ]


def test_main_categorizes_transactions_with_flattened_budget_categories(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    statement_path = tmp_path / "checking.ofx"
    statement_path.write_text(
        """
        <OFX><BANKMSGSRSV1><STMTTRNRS><STMTRS><CURDEF>USD
        <BANKACCTFROM><ACCTID>checking</ACCTID></BANKACCTFROM>
        <BANKTRANLIST><STMTTRN><TRNTYPE>DEBIT<DTPOSTED>20260427
        <TRNAMT>-7.00<FITID>checking-1<NAME>Coffee Shop<MEMO>CARD PURCHASE COF SHOP</STMTTRN></BANKTRANLIST>
        </STMTRS></STMTTRNRS></BANKMSGSRSV1></OFX>
        """,
        encoding="utf-8",
    )
    calls: list[str] = []

    def fake_enrich_transactions_with_brave_search(
        transactions: list[Transaction],
        *,
        request_delay_seconds: float | None = None,
    ) -> list[Transaction]:
        _ = request_delay_seconds
        calls.append("brave")
        return transactions

    def fake_categorize_transactions(transactions: list[Transaction]) -> list[Transaction]:
        calls.append("categorize")
        transactions[0].category_allocations = [
            CategoryAllocation(
                category_id="Food & Dining > Fast Food & Coffee",
                amount="-7.00",
                confidence=0.8,
                source=Source1.model,
            )
        ]
        return transactions

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ingest_statements.py",
            "--categorize",
            str(statement_path),
        ],
    )
    monkeypatch.setattr(
        ingest_statements,
        "enrich_transactions_with_brave_search",
        fake_enrich_transactions_with_brave_search,
    )
    monkeypatch.setattr(ingest_statements, "categorize_transactions", fake_categorize_transactions)

    main()

    assert calls == ["brave", "categorize"]
