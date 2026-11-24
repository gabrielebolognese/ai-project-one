/** Worker-thread entry: one data-parallel replica (see executor.ts). */
import { parentPort, workerData } from "node:worker_threads";
import { Replica, type WorkerInitData, type WorkerRequest, type WorkerResponse } from "./executor.js";

const init = workerData as WorkerInitData;
const replica = new Replica(init.cfg, init.params, init.grad, init.train, init.val);
const port = parentPort!;

port.on("message", (req: WorkerRequest) => {
  let res: WorkerResponse;
  try {
    if (req.type === "step") {
      res = { type: "step", id: req.id, loss: replica.step(req.offsets, req.T) };
    } else {
      const r = replica.evaluate(req.offsets, req.T);
      res = { type: "eval", id: req.id, lossSum: r.lossSum, count: r.count };
    }
  } catch (e) {
    res = { type: "error", id: req.id, message: e instanceof Error ? (e.stack ?? e.message) : String(e) };
  }
  port.postMessage(res);
});

port.postMessage({ type: "ready" } satisfies WorkerResponse);
