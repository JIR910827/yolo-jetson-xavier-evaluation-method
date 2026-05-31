const fs = require("fs");

const DEFAULT_RPC_URL = process.env.ETH_RPC_URL || "http://127.0.0.1:8545";
const DEFAULT_ABI_PATH =
  process.env.CONTRACT_ABI_PATH || "VisualRecordContract.abi.json";
const DEFAULT_ADDRESS_PATH =
  process.env.CONTRACT_ADDRESS_PATH || "VisualRecordContract.address.txt";

function parseArgs(argv) {
  const args = {};

  for (let i = 0; i < argv.length; i += 1) {
    const arg = argv[i];
    if (!arg.startsWith("--")) {
      continue;
    }

    const key = arg.slice(2);
    const next = argv[i + 1];
    if (!next || next.startsWith("--")) {
      args[key] = true;
    } else {
      args[key] = next;
      i += 1;
    }
  }

  return args;
}

function parseResult(resultText) {
  try {
    return JSON.parse(resultText);
  } catch {
    return {};
  }
}

function normalizeRecord(record) {
  const result = parseResult(record.Result || record[4] || "{}");

  return {
    Id: record.Id || record[0],
    DateTime: record.DateTime || record[1],
    PicHash: record.PicHash || record[2],
    PicId: record.PicId || record[3],
    Result: record.Result || record[4],
    DeviceId: record.DeviceId || record[5],
    Status: result.status || "",
    IsDefect: result.is_defect,
    ClassName: result.class_name || "",
    Confidence: result.confidence,
  };
}

function printTable(records) {
  const rows = records.map(normalizeRecord);

  console.table(
    rows.map((row) => ({
      Id: row.Id,
      Time: row.DateTime,
      Status: row.Status,
      Defect: row.IsDefect,
      Class: row.ClassName,
      Conf: row.Confidence,
      PicHash: row.PicHash,
      Device: row.DeviceId,
    }))
  );

  const okCount = rows.filter((row) => row.Status === "Ok").length;
  const failCount = rows.filter((row) => row.Status === "Fail").length;
  console.log(`Total: ${rows.length}, Ok: ${okCount}, Fail: ${failCount}`);
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const rpcUrl = args.rpc || args["rpc-url"] || DEFAULT_RPC_URL;
  const abiPath = args.abi || args["abi-path"] || DEFAULT_ABI_PATH;
  const addressPath =
    args.address || args["address-path"] || DEFAULT_ADDRESS_PATH;

  let Web3;
  try {
    Web3 = require("web3");
  } catch {
    throw new Error("Cannot load web3. Run `npm install web3@1.10.0` first.");
  }

  const abi = JSON.parse(fs.readFileSync(abiPath, "utf8"));
  const address = fs.readFileSync(addressPath, "utf8").trim();
  const web3 = new Web3(rpcUrl);
  const contract = new web3.eth.Contract(abi, address);

  if (args.id) {
    const record = await contract.methods.getVisualRecordByID(args.id).call();
    printTable([record]);
    if (args.json) {
      console.log(JSON.stringify(normalizeRecord(record), null, 2));
    }
    return;
  }

  const records = await contract.methods.getAllVisualRecords().call();
  printTable(records);

  if (args.json) {
    console.log(JSON.stringify(records.map(normalizeRecord), null, 2));
  }
}

main().catch((error) => {
  console.error(`Query failed: ${error.message}`);
  process.exitCode = 1;
});
