const fs = require("fs");
const path = require("path");
const crypto = require("crypto");
const { performance } = require("perf_hooks");

const DEFAULT_API_URL =
  process.env.BLOCKCHAIN_API_URL ||
  "http://<BLOCKCHAIN_API_HOST>:3000/iotBlockChain/CreateVisualRecord";
const DEFAULT_RPC_URL = process.env.ETH_RPC_URL || "http://127.0.0.1:8545";
const DEFAULT_DEVICE_ID = process.env.DEVICE_ID || "jetson-xavier-nx-0";
const DEFAULT_ABI_PATH =
  process.env.CONTRACT_ABI_PATH || "VisualRecordContract.abi.json";
const DEFAULT_ADDRESS_PATH =
  process.env.CONTRACT_ADDRESS_PATH || "VisualRecordContract.address.txt";

function pad2(value) {
  return String(value).padStart(2, "0");
}

function formatDateTime(date = new Date()) {
  const yyyy = date.getFullYear();
  const mm = pad2(date.getMonth() + 1);
  const dd = pad2(date.getDate());
  const hh = pad2(date.getHours());
  const mi = pad2(date.getMinutes());
  const ss = pad2(date.getSeconds());
  return `${yyyy}-${mm}-${dd} ${hh}:${mi}:${ss}`;
}

function hashFileSha256(filePath) {
  const hash = crypto.createHash("sha256");
  hash.update(fs.readFileSync(filePath));
  return hash.digest("hex");
}

function normalizeStatus(status) {
  const value = String(status || "Ok").trim().toLowerCase();
  if (["fail", "ng", "bad", "defect", "defective"].includes(value)) {
    return "Fail";
  }
  return "Ok";
}

function buildScrewResult({
  status = "Ok",
  confidence,
  className,
  boxes,
  extraResult = {},
}) {
  const normalizedStatus = normalizeStatus(status);
  const isFail = normalizedStatus === "Fail";

  return {
    product_type: "screw",
    status: normalizedStatus,
    class_name: className || normalizedStatus,
    is_defect: isFail,
    good_or_bad: isFail ? 0 : 1,
    quantity: 1,
    good: isFail ? 0 : 1,
    bad: isFail ? 1 : 0,
    confidence: confidence === undefined ? null : Number(confidence),
    boxes: boxes || [],
    ...extraResult,
  };
}

function buildVisualRecord({
  id,
  date = new Date(),
  picHash,
  picId,
  deviceId = DEFAULT_DEVICE_ID,
  result,
  status = "Ok",
  confidence,
  className,
  boxes,
  extraResult = {},
}) {
  const timestamp = Math.floor(date.getTime() / 1000);
  const resolvedId = id || `screw-${timestamp}`;
  const resolvedPicId = picId || resolvedId;
  const resolvedResult =
    result ||
    buildScrewResult({
      status,
      confidence,
      className,
      boxes,
      extraResult,
    });

  return {
    Id: resolvedId,
    DateTime: formatDateTime(date),
    PicHash: picHash || crypto.randomBytes(16).toString("hex"),
    PicId: resolvedPicId,
    Result: JSON.stringify(resolvedResult),
    DeviceId: deviceId,
  };
}

function toApiParams(record) {
  return {
    ID: record.Id,
    DateTime: record.DateTime,
    PicHash: record.PicHash,
    PicID: record.PicId,
    Result: record.Result,
    DeviceID: record.DeviceId,
  };
}

async function uploadByApi(record, apiUrl = DEFAULT_API_URL) {
  const url = new URL(apiUrl);
  const params = toApiParams(record);

  for (const [key, value] of Object.entries(params)) {
    url.searchParams.set(key, String(value));
  }

  const start = performance.now();
  const response = await fetch(url);
  const responseText = await response.text();
  const durationMs = Number((performance.now() - start).toFixed(3));

  return {
    mode: "api",
    ok: response.ok,
    status: response.status,
    durationMs,
    url: url.toString(),
    responseText,
    record,
  };
}

function readContractConfig({
  abiPath = DEFAULT_ABI_PATH,
  addressPath = DEFAULT_ADDRESS_PATH,
}) {
  const resolvedAbiPath = path.resolve(abiPath);
  const resolvedAddressPath = path.resolve(addressPath);

  return {
    abi: JSON.parse(fs.readFileSync(resolvedAbiPath, "utf8")),
    address: fs.readFileSync(resolvedAddressPath, "utf8").trim(),
    abiPath: resolvedAbiPath,
    addressPath: resolvedAddressPath,
  };
}

async function uploadByContract(record, options = {}) {
  let Web3;
  try {
    Web3 = require("web3");
  } catch (error) {
    throw new Error(
      "Cannot load web3. Run `npm install web3@1.10.0` in this folder first."
    );
  }

  const rpcUrl = options.rpcUrl || DEFAULT_RPC_URL;
  const { abi, address, abiPath, addressPath } = readContractConfig(options);
  const web3 = new Web3(rpcUrl);
  const accounts = await web3.eth.getAccounts();

  if (!accounts.length) {
    throw new Error("No unlocked account from RPC. Check geth --http and account unlock.");
  }

  const from = options.from || accounts[0];
  const contract = new web3.eth.Contract(abi, address);

  const start = performance.now();
  const receipt = await contract.methods
    .createVisualRecord(
      record.Id,
      record.DateTime,
      record.PicHash,
      record.PicId,
      record.Result,
      record.DeviceId
    )
    .send({
      from,
      gas: Number(options.gas || 3000000),
      gasPrice: String(options.gasPrice ?? "0"),
    });
  const durationMs = Number((performance.now() - start).toFixed(3));

  return {
    mode: "contract",
    ok: true,
    durationMs,
    rpcUrl,
    contractAddress: address,
    from,
    transactionHash: receipt.transactionHash,
    blockNumber: receipt.blockNumber,
    abiPath,
    addressPath,
    record,
  };
}

async function uploadRecognitionToChain(options = {}) {
  const imagePath = options.imagePath || options.image;
  const record = buildVisualRecord({
    ...options,
    picHash: imagePath ? hashFileSha256(imagePath) : options.picHash,
  });

  if ((options.mode || "contract") === "api") {
    return uploadByApi(record, options.apiUrl);
  }

  return uploadByContract(record, options);
}

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

function parseBoxes(rawBoxes) {
  if (!rawBoxes) {
    return [];
  }
  if (Array.isArray(rawBoxes)) {
    return rawBoxes;
  }
  return JSON.parse(rawBoxes);
}

async function runCli() {
  const args = parseArgs(process.argv.slice(2));
  const result = await uploadRecognitionToChain({
    mode: args.mode || "contract",
    rpcUrl: args.rpc || args["rpc-url"],
    apiUrl: args.url || args["api-url"],
    abiPath: args.abi || args["abi-path"],
    addressPath: args.address || args["address-path"],
    from: args.from,
    gas: args.gas,
    gasPrice: args["gas-price"],
    id: args.id,
    picId: args["pic-id"],
    picHash: args["pic-hash"],
    imagePath: args.image || args["image-path"],
    deviceId: args.device || args["device-id"],
    status: args.status,
    confidence: args.confidence,
    className: args.class || args["class-name"],
    boxes: parseBoxes(args.boxes),
  });

  console.log(`Mode: ${result.mode}`);
  console.log(`Record ID: ${result.record.Id}`);
  console.log(`Device ID: ${result.record.DeviceId}`);
  console.log(`PicHash: ${result.record.PicHash}`);
  console.log(`Result: ${result.record.Result}`);
  console.log(`Duration: ${result.durationMs} ms`);

  if (result.mode === "contract") {
    console.log(`RPC: ${result.rpcUrl}`);
    console.log(`Contract: ${result.contractAddress}`);
    console.log(`From: ${result.from}`);
    console.log(`Transaction hash: ${result.transactionHash}`);
    console.log(`Block number: ${result.blockNumber}`);
  } else {
    console.log(`URL: ${result.url}`);
    console.log(`Status: ${result.status}`);
    console.log(`Response: ${result.responseText}`);
  }

  if (!result.ok) {
    process.exitCode = 1;
  }
}

if (require.main === module) {
  runCli().catch((error) => {
    console.error(`Upload failed: ${error.message}`);
    process.exitCode = 1;
  });
}

module.exports = {
  buildScrewResult,
  buildVisualRecord,
  toApiParams,
  uploadByApi,
  uploadByContract,
  uploadRecognitionToChain,
};
