# Neural Networks on FPGAs — Study Notes

An asynchronous study guide on embedded AI and neural network deployment on FPGAs.

---

## Why Not Just Use a GPU?

When you train a neural network, you typically do it on a GPU. GPUs have large memory banks, massive parallelism, and native support for floating-point arithmetic — everything a training workload needs.

FPGAs are a completely different environment. You do not have the luxury of taking a full-blown model and dropping it onto the chip. Even very simple models require careful adaptation. **On an FPGA, everything is about optimization.**

That said, FPGAs offer real advantages for inference at the edge:

- Deterministic, ultra-low latency
- Low power consumption compared to GPUs
- Fully customizable parallelism in hardware
- No operating system dependency — the logic runs directly on silicon

### A Note on Parallelism

Parallelism means doing multiple things at exactly the same time — not switching between them quickly, but literally simultaneously.

A CPU executes instructions one after another (or a handful at a time with multiple cores). A GPU improves on this with thousands of smaller cores that can each run the same operation on different data simultaneously — which is why matrix multiplications in neural networks map so well to GPU hardware.

An FPGA takes a different approach entirely. Instead of running a program on general-purpose cores, you **build the computation directly into the chip's wiring**. Every operation you need has its own dedicated piece of hardware, and all of those pieces can run at the same time, every clock cycle.

Think of it this way:

```
CPU:   [multiply] → [multiply] → [multiply] → [multiply]   (sequential)
GPU:   [multiply]   [multiply]   [multiply]   [multiply]    (parallel, shared cores)
FPGA:  [multiply]   [multiply]   [multiply]   [multiply]    (parallel, dedicated circuits)
        ↑ neuron 1   ↑ neuron 2   ↑ neuron 3   ↑ neuron 4
```

On an FPGA, if your model has 64 neurons in a layer, you can instantiate 64 multipliers in hardware and compute all of them in a single clock cycle. There is no scheduling, no memory bus contention, no core sharing. The parallelism is structural — it exists in the physical layout of the chip.

---

## The Physical Anatomy of an FPGA

An FPGA is not a general-purpose processor. It is a chip filled with a large array of configurable building blocks, connected by a programmable routing fabric. Understanding what each block does — and why it exists — is essential for reasoning about what fits on a chip and what doesn't.

Here is an overview of the main physical resources you will encounter.

---

### LUT — Look-Up Table

The LUT is the fundamental logic building block of an FPGA. Every piece of combinational logic you can imagine — AND gates, OR gates, XOR gates, multiplexers, comparators — is implemented using LUTs.

A LUT with N inputs is essentially a small truth table stored in memory. It has 2ᴺ possible input combinations, and for each one, it stores a pre-computed output bit. When signals arrive at the inputs, the LUT looks up the corresponding output instantly.

```
4-input LUT — 2⁴ = 16 stored output bits

Inputs: A B C D
           ↓
   [ truth table in memory ]
           ↓
        Output: 1 bit
```

Because the truth table is programmable, a single LUT can implement any logic function of its inputs. A 4-input LUT can be an adder one day and a comparator the next — it depends entirely on what the bitstream writes into its memory cells.

LUTs are what get consumed when your HDL code describes logic. An adder uses LUTs. A multiplexer uses LUTs. A decoder uses LUTs. When a synthesis report says "you are using 70% of LUT resources," it means 70% of those truth-table blocks are occupied.

---

### FF — Flip-Flop

A Flip-Flop (also called a register) stores a single bit of state. It captures the value on its input at the moment of a clock edge and holds it steady until the next clock edge.

```
        clock edge
            ↓
Input ──► [ FF ] ──► Output (held stable until next clock)
```

LUTs compute things. Flip-Flops remember things — for exactly one clock cycle at a time.

In a neural network context, FFs are used to pipeline results between stages. If you have multiple layers of computation, each layer's output can be registered into FFs so the next layer can start processing while the previous one is already accepting new inputs. This is **pipelining**, and it is how FPGAs achieve high throughput alongside low latency.

LUTs and FFs almost always appear together. On most FPGAs, they are physically paired inside a structure called a **Slice** or **Logic Element** — one LUT and one or more FFs sharing the same physical cell. This pairing exists because combinational logic (LUT) and state storage (FF) are almost always used together.

---

### DSP Block — Digital Signal Processor

DSP blocks are hardened arithmetic units built directly into the silicon — unlike LUTs, which implement logic by lookup, DSP blocks contain actual multiplier and accumulator circuits etched into the chip.

A typical DSP block can perform:

```
result = (A × B) + C
```

in a single clock cycle, on operands of 18 to 27 bits depending on the chip family.

This matters enormously for neural networks. The core operation of every neuron is a **multiply-accumulate**: multiply each input by its weight, then sum the results. A DSP block does exactly this, far more efficiently than building the same operation out of LUTs.

```
Neuron output = Σ (input[i] × weight[i])
                       ↑
              This is what DSP blocks are built for
```

When you quantize your model to INT8 or INT16, the operands fit neatly into DSP block inputs, and you can pack more operations per block. When weights are `float32`, you either need to chain multiple DSPs together or fall back to LUTs — both of which are more expensive. This is one of the concrete reasons quantization matters on FPGAs.

---

### BRAM — Block RAM

BRAM is dedicated on-chip memory. Unlike external DRAM (which is off-chip, slow, and energy-hungry), BRAM sits directly on the FPGA die and can be read or written in a single clock cycle.

Each BRAM block is a small, dual-port memory — typically 18 Kb or 36 Kb in size. Dual-port means it has two independent access ports, so two different parts of your circuit can read from or write to it simultaneously.

In a neural network on an FPGA, BRAM is used to store:

- **Weights** — the parameters of the network, loaded from the bitstream or from external memory at startup
- **Activations** — intermediate values between layers, especially in pipelined designs
- **Input buffers** — staging area for incoming data before it enters the computation

The critical constraint is that BRAM is scarce. A mid-range FPGA might have a few megabytes of total BRAM. A neural network with millions of parameters simply will not fit. This is precisely why quantization (smaller weights) and pruning (fewer weights) are not optional — they are the difference between a model that fits and one that does not.

```
float32 weight × 1M parameters = 4 MB  →  likely does not fit
int8 weight    × 1M parameters = 1 MB  →  might fit
int8 weight    × 100K parameters (pruned) = 100 KB  →  fits comfortably
```

---

### Programmable Interconnect

The blocks above — LUTs, FFs, DSPs, BRAMs — are arranged in a fixed grid across the chip. What makes an FPGA reconfigurable is the **programmable interconnect**: a dense mesh of wires running horizontally and vertically between all resource blocks, with configurable switches at every intersection.

The bitstream programs these switches to either connect or disconnect, effectively routing signals between whatever blocks your design needs to communicate. This is the physical substrate that Place & Route operates on.

The interconnect is fast, but not free. Every switch a signal passes through adds a small delay. Long routes across the chip accumulate more delay than short, local routes. This is why the placement decisions made during Place & Route directly impact how fast your circuit can run.

---

### How They Work Together in a Neural Network

A single layer of a neural network on an FPGA uses all of these resources in concert:

```
BRAM          →  stores the layer's weights
DSP blocks    →  compute weight × input for each neuron
LUTs          →  implement the activation function (e.g. ReLU, comparisons)
Flip-Flops    →  pipeline the output to the next layer
Interconnect  →  carries signals between all of the above
```

The synthesis and place & route tools are responsible for mapping your high-level description onto this physical substrate, respecting the available count of each resource type. When a design fails to fit, it is usually because one resource — most commonly BRAM or DSP blocks — has been exhausted.

---

## The Core Challenge: Constraint vs. Capability

The fundamental tension in embedded AI is this: machine learning models are designed in environments with abundant resources, but deployed in environments where every bit of memory, every logic block, and every clock cycle is scarce.

This is what makes FPGA-based AI different from cloud inference or even microcontroller deployment. You are not running software on top of hardware — you are **turning the model itself into hardware**.

---

## Optimization Techniques

Before a neural network can run on an FPGA, it must be compressed and restructured. The two most important techniques are quantization and pruning.

### Quantization

Neural networks are typically trained using 32-bit floating-point numbers (`float32`). These are precise but expensive — in terms of memory, bandwidth, and the size of the arithmetic units required to compute them.

Quantization reduces this precision to smaller integer representations:

```
float32  →  4 bytes per weight
int8     →  1 byte per weight    (4× smaller)
int4     →  0.5 bytes per weight (8× smaller)
```

Fewer bits means:
- Smaller memory footprint
- Less silicon area on the FPGA
- Faster arithmetic operations in hardware

The accuracy loss from quantization, when done carefully, is minimal. The model learns to compensate during a quantization-aware training step.

### Pruning

Not all weights in a neural network contribute equally to its output. Many weights end up close to zero — they could be removed without meaningfully affecting predictions.

Pruning identifies and removes these near-zero weights, producing a **sparse network**:

- Fewer multiplications to perform
- Zero-weight connections can be eliminated entirely in hardware
- The resulting model is smaller and faster without retraining from scratch

Pruning and quantization are often applied together for maximum compression.

---

## From Python to Hardware: You Don't Write HDL by Hand

At this point you have a trained, optimized model — quantized, pruned, and ready for deployment. The natural next question is: how does it actually become a circuit?

The naive assumption would be that you now have to manually write VHDL or Verilog — describing every multiplier, every adder, every register by hand. In practice, that is not how modern embedded AI workflows operate.

Instead, you stay in Python.

Specialized frameworks exist that take a Keras or PyTorch model as input and automatically generate the RTL (Register-Transfer Level) code that describes the equivalent hardware circuit. You provide the model and the target board; the tool produces synthesizable HDL.

```
Python (Keras / PyTorch model)
          │
          │  framework converts model to RTL
          ▼
RTL code (VHDL / Verilog)  ←── generated automatically
          │
          ▼
     HDL Pipeline (synthesis, place & route, bitstream)
```

This is significant for a few reasons:

- You do not need to be an HDL expert to deploy a neural network on an FPGA
- The framework applies optimizations (loop unrolling, quantization-aware arithmetic, pipelining) automatically during the conversion
- The generated RTL is correct-by-construction for the target chip's resource constraints

### The Trade-offs of Not Writing HDL

This convenience comes with real costs. Understanding them helps you decide when the automatic path is good enough and when it isn't.

**Resource inefficiency**

A framework generating RTL from a Python model does not know your specific use case — it generates conservative, general-purpose hardware. A human HDL engineer would look at the same model and find opportunities to share resources, reuse logic, or exploit chip-specific features that the tool will never attempt. The automatic output typically uses more LUTs, DSPs, and BRAM than a hand-optimized implementation of the same model.

On a large FPGA with room to spare, this does not matter. On a constrained chip where every block counts, it can be the difference between a design that fits and one that does not.

**Loss of timing control**

When you write HDL by hand, you control exactly when every signal is computed and registered. You can insert pipeline stages precisely where timing is tight, and collapse them where latency matters more than throughput.

Auto-generated RTL applies a fixed pipelining strategy. If the generated circuit fails timing during place & route, your only option is usually to adjust high-level parameters and regenerate — you cannot easily reach into the RTL and fix a specific path.

**Opacity**

The generated VHDL or Verilog can be thousands of lines long and extremely difficult to read. If something goes wrong — wrong outputs, timing failures, unexpected resource usage — debugging the generated code is hard. You are no longer looking at something a human wrote with intent; you are reading the output of an automated transformation.

With hand-written HDL, every line has a reason. You can trace a bug back to a design decision.

**Portability**

Auto-generated RTL is often tightly coupled to a specific FPGA family or vendor toolchain. Moving the same model to a different chip may require re-running the entire conversion pipeline and accepting that the output will look completely different. Hand-written HDL, written carefully, can be more portable across vendors.

**When the automatic path is the right choice**

Despite these costs, the Python-to-RTL workflow is the right starting point for most embedded AI projects:

- Prototyping and validation — proving the model works on hardware before investing in optimization
- Projects where the FPGA has sufficient resources and the automatic output fits comfortably
- Teams where machine learning expertise is available but HDL expertise is not
- Time-constrained deployments where getting to working hardware quickly outweighs peak efficiency

Hand-written HDL becomes worthwhile when the design must run on a very constrained chip, when maximum throughput or minimum latency is a hard requirement, or when the generated output consistently fails to meet timing.

This Python-to-RTL step is what connects the machine learning world to the hardware world — it is the bridge that makes FPGA-based AI accessible without requiring two separate engineering disciplines. Knowing its limits is what lets you use it wisely.

---

## From HDL to a Working System: Vitis HLS and Vivado in Practice

The sections above describe the HDL pipeline conceptually — HDL in, bitstream out. In practice, that pipeline is split across two distinct tools, and getting from "I have a synthesized neural network circuit" to "I have a working chip" involves a few more concrete pieces than the abstract diagram shows.

### Two forms of the same circuit

Once your model becomes HDL, that HDL exists in two usable forms, not one:

- **A textual description** — the actual `.v` / `.vhd` files, edited like any other source code
- **A packaged, connectable block** — the same circuit wrapped with a standardized interface, so it can be dropped onto a visual canvas and wired to other blocks without touching its internals

This packaged form is called an **IP core** (Intellectual Property core). It's not a different technology — it's the same circuit, prepared for reuse and assembly rather than line-by-line editing.

### Why a visual assembly step exists at all

A synthesized neural network is useless in isolation. On a real SoC-FPGA (like the Zybo or ZCU boards), it needs: a way to receive data from outside the chip, a way for the on-board ARM processor to start it and read results back, and shared clock/reset wiring with everything else on the chip. Writing all of that connective "glue" logic by hand for every project would be repetitive and error-prone.

Instead, Xilinx provides a canvas tool where pre-built IP cores are dropped in and connected by drawing wires between labeled pins.

- **IP Integrator** — the canvas tool itself
- **Block Design** — the diagram you build in it (in Vivado, not Vitis HLS)

Critically, a block in this diagram might come from two different sources — and the system treats them identically:

- Generated automatically by Vitis HLS from your C++ (hls4ml's output)
- Written by hand in Verilog/VHDL and manually packaged into an IP core (via *Tools → Create and Package New IP* in Vivado)

Hand-written HDL is worth packaging this way when the design needs custom control logic that doesn't emerge naturally from C++, when two IP cores don't quite fit together and need small "glue" logic between them, or when hand-optimized RTL genuinely outperforms what HLS can generate — the same trade-off already discussed under "The Trade-offs of Not Writing HDL."

### AXI: the shared language that lets blocks be connected at all

If every design invented its own private way for two blocks to exchange data, no IP core from one project could ever be reused in another. AXI (Advanced eXtensible Interface — ARM's bus standard, adopted by Xilinx as the default for IP-to-IP communication) solves this by standardizing the pins and rules every block uses to talk to its neighbors. Two protocols cover the two situations that come up constantly:

**AXI-Lite — occasional control commands**
Used for simple register-style reads and writes: "start now," "what's your status," "read this configuration value." Doesn't need to move much data, just needs to be simple and addressable.

**AXI-Stream — continuous, one-directional data flow**
Used when data flows continuously from one block to another (pulse samples into a classifier, results out of it). The core problem it solves: the sender and receiver don't necessarily run at the same rate, so there needs to be an agreed way for either side to say "wait." This is done with a two-signal handshake:

```
TVALID (sender)   →  "the data on my output right now is valid"
TREADY (receiver) →  "I'm ready to accept data"

Transfer happens only on a clock edge where BOTH are high.
```

A third signal, `TLAST`, marks the final sample of a packet when data arrives in discrete batches rather than an endless stream. These are exactly the `input_r_TVALID` / `input_r_TREADY` / `input_r_TLAST` pins that appear automatically on any hls4ml-generated block's input — they aren't custom names, they're the standard AXI-Stream handshake attached by the tool.

### FIFOs: buffering for rate mismatches

Even with a working handshake, data often arrives in irregular bursts from outside the chip while a compute block wants a smooth, steady stream. Something needs to hold data that has arrived but hasn't been consumed yet.

**FIFO** (First-In-First-Out buffer) — a small queue, implemented in BRAM, where the first sample written is the first one read out. This is the practical role of a ComBlock-style IP core in a typical block design: it buffers incoming samples in a FIFO before releasing them, via AXI-Stream, into the classifier.

### The HLS control protocol: how a block reports its own status

Since a synthesized block takes multiple clock cycles to compute a result (see pipelining, above), it needs a way to signal its own state to the rest of the system — separate from the data-stream handshake. Vitis HLS attaches this automatically to every function it synthesizes:

| Signal | Meaning |
|---|---|
| `ap_start` | External signal: "begin processing now" |
| `ap_done` / `..._ap_vld` | The block signaling: "finished — this output is now valid" |
| `ap_idle` | The block signaling: "not currently processing, safe to restart" |

This is why a classification output appears as a pair — e.g. `result[31:0]` alongside `result_ap_vld` — rather than the number alone: the system also needs to know the exact cycle at which that number became trustworthy, since it isn't valid on every cycle.

### Three verification checkpoints, cheapest to most expensive

Because synthesis and place & route are slow, Vitis HLS provides three checkpoints before committing to that cost, each catching a different class of bug:

1. **C Simulation** — runs the C++ testbench against the C++ design in plain software. Confirms the algorithm itself is correct, before any hardware exists.
2. **C Synthesis** — generates the actual RTL (Verilog/VHDL) from the C++. Expensive, but still just produces a circuit description — it doesn't yet prove that circuit behaves like the original C++.
3. **C/RTL Cosimulation** — re-runs the *same* testbench, but now against the generated RTL, and compares outputs to the C++ version. Catches subtle bugs synthesis can introduce (fixed-point truncation effects, timing assumptions) that C Simulation alone can't see.

Only after all three pass does it make sense to move into Vivado for full block-design integration and implementation.

### Putting it together

```
C++ (Vitis HLS)
   │  C Simulation → C Synthesis → C/RTL Cosimulation
   ▼
IP core (RTL, packaged with AXI-Lite control + AXI-Stream data pins)
   │
   ▼
Block Design (Vivado IP Integrator)
   — wire IP core to FIFO buffers, AXI interconnect, ARM processing system —
   │  Generate Block Design → auto-generated top-level wrapper (do not hand-edit)
   ▼
Synthesis → Implementation → Bitstream
```

A practical note on Mac/Apple Silicon: everything up to and including "generate RTL" (`hls_model.write()` in hls4ml, C Simulation) runs as plain Python/C++ and works natively on macOS/ARM. Only C Synthesis, C/RTL Cosimulation, and everything inside Vivado require the vendor toolchain, which AMD/Xilinx ships only for Windows and x86-64 Linux — never macOS, and never ARM. A workable split is: iterate on the model and generate the HLS project locally, then sync that project folder to a remote x86 Linux machine (lab workstation or cloud instance) to run synthesis and Vivado integration.

---

## From Model to Hardware: The HDL Pipeline

Once the model is converted to RTL, it enters the standard FPGA compilation pipeline. This process goes through several stages.

```
HDL Code (VHDL / Verilog)
        │   loop unrolling applied
        ▼
High-Level Synthesis
        │   converts logic description into circuit netlist
        ▼
Place & Route
        │   maps netlist onto physical LUTs, FFs, and DSPs
        ▼
Bitstream  (.bit / .rbf / .sof)
        │   compiled binary — encodes every switch and connection
        ▼
Programmed onto the FPGA
```

### HDL and Loop Unrolling

Hardware Description Languages like VHDL and Verilog describe digital circuits, not sequential programs. One key technique when mapping neural network computations to HDL is **loop unrolling**: instead of executing a loop iteration by iteration, each iteration becomes independent parallel hardware.

```
// Sequential (software)
for i in range(4):
    sum += weight[i] * input[i]

// Unrolled (hardware — all computed simultaneously)
sum = w0*x0 + w1*x1 + w2*x2 + w3*x3
```

This is what gives FPGAs their latency advantage: operations that would take N clock cycles sequentially can happen in a single cycle when unrolled.

### Synthesis

High-level synthesis tools translate the HDL description into a **circuit netlist** — a representation of logic gates and their connections. This step applies optimizations like resource sharing, pipelining, and timing analysis.

### Place & Route

Place & Route is the step where an abstract circuit description becomes a physical layout on a specific chip. It is one of the most computationally intensive steps in the FPGA toolchain, and understanding it explains a lot about why FPGA compilation takes minutes or hours rather than seconds.

**What the netlist contains at this point**

After synthesis, you have a netlist: a list of logical components (a multiplier here, an adder there, a register somewhere else) and the connections between them. It describes *what* the circuit does, but not *where* it lives on the chip.

**Placement**

The FPGA die contains thousands to millions of physical resource blocks arranged in a fixed grid — LUTs, Flip-Flops, DSPs, BRAMs. Placement is the process of deciding which logical component from the netlist gets assigned to which physical block on that grid.

This matters because the chip is finite and its blocks are fixed in location. A multiplier from your neural network layer must be mapped to a DSP block that actually exists on the chip, in a real physical position.

```
Netlist (abstract)          FPGA die (physical grid)
─────────────────           ──────────────────────────────
Multiplier A         →      DSP block at column 4, row 12
Adder B              →      LUT cluster at column 5, row 12
Register C           →      Flip-Flop at column 5, row 13
```

A good placement groups elements that communicate frequently close together on the die, minimizing the distance signals must travel.

**Routing**

Once every element has a position, routing connects them using the FPGA's programmable interconnect — a grid of wires and configurable switches that run between the resource blocks.

This is where the tool must literally find physical wire paths through the chip for every connection in your netlist. Two elements might need to be connected but the most direct wire path is already occupied by another signal. The router has to find an alternative path, which may be longer and therefore slower.

**Why wire length matters: timing**

Signals in digital circuits travel fast, but not instantaneously. A longer wire introduces more propagation delay. If the path between two elements is too long, the signal might not arrive before the next clock edge — and the circuit produces wrong results.

The place & route tool runs **timing analysis** continuously, checking whether every connection meets the required timing constraints. If something fails, it tries a different placement or routing path and checks again. This iterative process is why it takes so long.

```
Short path  →  fast signal  →  timing met   ✓
Long path   →  slow signal  →  timing fail  ✗  → tool reroutes and retries
```

**The output: a fully specified physical layout**

When place & route succeeds, every logical element has a physical location, every connection has a physical wire path, and every timing constraint has been verified. This complete specification is what gets encoded into the bitstream.

### Bitstream

The final output is a **bitstream** — a binary file that encodes the complete configuration of the FPGA: every LUT function, every connection, every register. Loading this file onto the chip transforms it into a dedicated inference accelerator.

| Extension | Vendor              |
|-----------|---------------------|
| `.bit`    | Xilinx / AMD        |
| `.rbf`    | Intel / Altera      |
| `.sof`    | Intel (JTAG target) |

---

## Inference on an FPGA

Inference is the AI doing its actual work — taking new inputs and producing predictions. This is fundamentally different from the training phase.

A few important distinctions from the LLM world:

- The model is **fixed at compile time** — weights are baked into the bitstream
- There is no memory of previous inputs between runs
- Latency is measured in **microseconds**, not seconds
- No GPU, no operating system, no runtime interpreter — just logic on silicon

In practice, inference on an FPGA looks like feeding data into a hardware IP core and reading back a result. The neural network has become a circuit.

### Training vs. Test Data

Even in the embedded context, the principle of separating data applies:

- **Training data** is used to learn the model's weights
- **Test data** is held back and only used to evaluate final accuracy

This separation is especially important when validating a quantized or pruned model, since compression can introduce small accuracy losses that only become visible on unseen data.

---

## Case Study: Real-Time vs. Offline Signal Processing — EMG on FPGAs

Everything above describes what happens *after* a model has been trained and a fixed input representation has been decided. But for biosignals like EMG, the path from raw sensor data to that input representation is itself full of decisions that behave very differently offline (in research, on a full recorded dataset) versus online (streaming, on an FPGA, with no access to the future). This section works through those differences concretely, since they directly shape what you can and cannot put in the bitstream.

### The core distinction: causality, not "how much data"

It's tempting to think the difference between offline and real-time processing is about *how much* signal you get to look at — a full recording vs. a short window. That's not quite right. The real dividing line is **causality**: whether a computation is only allowed to use samples from the past, or whether it's also allowed to use samples from the future.

- A **zero-phase filter** (e.g. `filtfilt`, common in offline EMG research) runs the filter forward and then backward through the entire signal. To produce the filtered value at sample *t*, it uses information from samples that come *after t* in the recording. This is impossible in real time — you don't have "the future" of a signal that is still arriving.
- A **feature computed over a window** (e.g. RMS over the last 150 ms) only needs a buffer of *past* samples. This is fully causal: at any instant *t*, you look backward at the last *L* samples already received. It introduces latency (you have to wait for the window to fill), but it never needs data that hasn't happened yet.

So the FPGA-relevant reframing is: **offline research tools are often non-causal by default, and porting them to hardware means finding the causal equivalent — not simply "less data."**

### Filtering: same necessity, different implementation

Band-pass and notch filtering (removing power-line interference and out-of-band artifacts) are not optional in either setting — the noise doesn't go away because the system is streaming. What changes is the filter *structure*:

| | Offline | Real-time / FPGA |
|---|---|---|
| Filter type | Zero-phase (filtfilt), non-causal | Causal IIR (e.g. direct-form Butterworth) or FIR with fixed delay |
| Data required | Entire recording | Rolling buffer of past samples |
| Phase response | Linear phase (no distortion) | IIR introduces phase distortion; FIR can be linear-phase at the cost of more taps/latency |
| FPGA resources | N/A (done offline in software) | Maps to DSP blocks (multiply-accumulate for each tap/coefficient) and BRAM (buffering samples) |

This is a direct case of the "Constraint vs. Capability" tension described earlier in this guide: the offline method assumes unlimited access to the whole signal, while the hardware target has a fixed, small buffer and must produce an output every clock cycle without waiting for data that hasn't arrived yet.

### Feature extraction: causal, but not free — it costs latency

Classical EMG features (RMS, MAV, waveform length) are inherently window-based, which makes them naturally compatible with real-time operation — but the window size becomes a **fixed latency budget**, not just a hyperparameter chosen for accuracy.

- Offline research typically picks a window size purely to maximize decoding accuracy (e.g. a sensitivity analysis might find 150 ms optimal for a given task).
- Real-time systems must additionally respect a latency ceiling that the control application tolerates (often cited around 300 ms end-to-end in myoelectric control literature) — so the "best" window offline may not be usable online if it doesn't leave enough budget for the rest of the pipeline (filtering, inference, actuation).

On an FPGA specifically, this maps onto a **circular buffer implemented in BRAM**: each new sample overwrites the oldest one, and features are ideally computed *incrementally* (e.g. maintaining a running sum of squares for RMS and updating it by removing the outgoing sample and adding the incoming one) rather than recomputing the full window from scratch every cycle. Incremental computation is what keeps the DSP/LUT cost per new sample constant instead of scaling with window length.

### Formalizing the latency budget: analysis window (Tₐ) and estimation time (Tₑ)

The causal-window idea above can be made precise with two measurable quantities, used in a real deployed HD-sEMG/FPGA prosthetic control platform (Molinari et al., *A Wearable Platform for Real-Time Control of a Prosthetic Hand by High-Density EMG*, medRxiv, 2025 — a Zynq UltraScale+ MPSoC system processing 128-channel HD-sEMG in real time):

- **Tₐ (analysis window)** — how much signal, in milliseconds, must be accumulated before a single prediction can be made. This is the same "window size" discussed above (e.g. 30, 60, 90, 120, 150 ms), just given a formal name.
- **Tₑ (estimation time)** — how long the hardware actually takes to process that window end-to-end (every pipeline stage: conditioning filters, any adaptation/artifact-mitigation step, feature extraction, model inference, and output transmission) and produce a result.

Two derived quantities matter more than either number alone:

- **Tₐ + Tₑ** — the accumulated time from the start of the acquisition window to the end of processing; this is the true end-to-end latency a user or downstream controller experiences.
- **Tₑ / Tₐ** — the ratio that indicates the degree of overlap achievable between consecutive windows:
  - **Ratio > 1** — processing takes longer than the window's own duration, so the system cannot start the next window before finishing the current one. No overlap is possible, and there's a real risk of missing transient signal information between windows.
  - **Ratio < 1** — processing finishes faster than the window duration, leaving room for consecutive windows to overlap (the classic 50%-overlap pattern seen elsewhere in EMG literature). Overlap improves the *effective* temporal resolution of the output without shrinking Tₐ itself.

**This framework applies identically whether the model is classical ML or deep learning.** Tₐ is simply "how much raw signal the model's input requires," and Tₑ is "how long the hardware takes to turn that input into an output" — neither concept is specific to hand-crafted features. A CNN reading a raw or lightly-processed window still has a Tₐ (the window it consumes) and a Tₑ (its inference latency on the target FPGA); the numbers typically differ from a classical pipeline — Tₑ tends to grow with model depth and parameter count, per the DSP/BRAM cost discussion above — but the same ratio, and the same overlap-vs-no-overlap interpretation, still governs whether the deployed system can keep up with incoming data.

**Practical implication for candidate comparison:** Tₑ is a natural, hardware-measured metric to report per candidate model (alongside LUT/DSP/BRAM/power) when comparing architectures on an FPGA target — it directly answers "does this model's real-time performance let me overlap windows, or does it force me to process sequentially with gaps," which is a concrete, physically-verifiable complement to the offline accuracy-vs-resource-cost curve described later in this guide.

### Classical ML vs. deep learning: not a real-time filter, a resource trade-off

A common misconception is that "real-time" rules out either classical ML or deep learning. It doesn't — both are used in deployed real-time biosignal systems. What differs is where each spends its resource budget:

- **Classical ML on hand-crafted features** (Ridge regression, small MLPs, etc.) keeps two separable, lightweight stages: causal feature extraction (cheap, incremental) followed by a small, often closed-form or low-depth model. Linear models in particular map to very few DSP blocks and have deterministic, low latency — attractive when BRAM/DSP budget is tight.
- **Deep learning on raw or lightly-processed signal** skips manual feature engineering, letting convolutional layers learn the representation — but this shifts cost into more MAC operations, more weights to store in BRAM, and a deeper pipeline, which is exactly the "million-parameter model doesn't fit" problem described earlier in this guide. Quantization and pruning become more important, not less, when the input is a continuous biosignal stream rather than a static dataset.

Either approach still needs a defined, causal windowing strategy at the input — the choice of window size is a shared constraint, not something deep learning bypasses.

### Other real-time vs. offline differences worth tracking

Beyond filtering and feature extraction, several other aspects of an EMG pipeline behave differently once the assumption of "a complete, static recording" is dropped:

- **Output smoothing**: offline studies often low-pass filter the *predicted* trajectory with a zero-phase filter too — this needs a causal replacement (e.g. an exponential moving average) in deployment.
- **Normalization/calibration**: offline normalization (z-score, MVC-based scaling) is often computed using statistics from the entire session. Online, only a short calibration window is available before normal operation starts, so parameters must be frozen or updated with a causal running estimate.
- **Electrode shift and signal drift**: offline work can note this as a limitation; a deployed system has to actively handle it (recalibration, adaptive/incremental model updates) since performance degrades over the course of continuous use.
- **Train/test split assumptions**: an offline R² reported on a static held-out set doesn't capture whether a model needs daily recalibration or generalizes across sessions — a genuinely different evaluation question for a system meant to run continuously.
- **Missing samples / jitter**: invisible in a clean recorded dataset, but a real concern in streaming acquisition (dropped samples, buffer overruns), requiring an explicit policy (interpolate, hold-last-value, flag) that offline processing never has to define.
- **Evaluation metrics**: offline research reports R², RMSE, correlation on a fixed test set; a deployed system is additionally judged on end-to-end latency and perceived responsiveness — metrics that only exist once the pipeline is actually running continuously.

### Why the offline research still matters

None of this makes offline analysis (like a sensitivity study over filter parameters, window sizes, or feature sets on a recorded dataset) pointless for an FPGA project — quite the opposite. It establishes the **accuracy ceiling** and identifies **which resources actually matter** (e.g. which spatial regions of a sensor array carry most of the useful information, or which feature set gives comparable accuracy at lower computational cost) before you commit anything to a bitstream. The real-time constraints discussed here don't replace that analysis; they define the second half of the trade-off space — latency, causality, and hardware resources — that the offline numbers alone can't tell you.

### The offline reference standard: a ceiling, not a target to replicate

It's worth being explicit about *why* the offline benchmark matters, because it's easy to either dismiss it (since it ignores hardware constraints entirely) or over-rely on it (assuming the offline-best model is automatically the right one to implement). Neither is correct.

The offline benchmark's real job is to establish **the best accuracy achievable on the task with no hardware constraints applied at all** — a ceiling. Without it, there's no way to say whether a given FPGA implementation is "good": good compared to what? A resource-efficient but inaccurate hardware model is meaningless without knowing how much accuracy was actually available to begin with, and how much of it was traded away for efficiency.

**The best architecture offline is not necessarily the best architecture on FPGA — and that mismatch is itself the interesting research question**, not a side note to work around. Concretely, in a study like the one discussed earlier in this guide:

- A larger MLP wins on offline accuracy, but costs the most DSP blocks, BRAM, and power on hardware, and tends to have the least deterministic latency of the group.
- A linear model (e.g. Ridge) loses on offline accuracy — it can't capture the same nonlinearity — but maps to closed-form, extremely cheap hardware: minimal DSP usage, near-instant and highly deterministic latency.
- Some intermediate architecture may offer the best **accuracy-per-DSP** or **accuracy-per-watt**, which is a different notion of "best" than the offline leaderboard shows, and only becomes visible once resource and power measurements are taken into account.

This is the same theme that motivates tools like hls4ml and resource-aware architecture search in embedded ML more broadly: accuracy rankings measured on a GPU/CPU do not transfer directly to rankings measured in LUTs, DSPs, BRAM, latency, or watts.

**Methodology this suggests:**

1. Establish the offline reference: accuracy of each candidate model (e.g. a linear model, a small MLP, a lightweight CNN) on the dataset, with no hardware constraints — this is the ceiling.
2. Implement each candidate on the FPGA (via HLS, and optionally hand-written HDL for at least one, to compare implementation paths).
3. Measure, per model: resource usage (LUT/DSP/BRAM), latency, power, and accuracy *after* hardware constraints such as quantization (which typically drops somewhat from the offline float32 number).
4. Plot accuracy against resource cost (or against power) across all candidate models — this trade-off curve is the actual deliverable, and it directly answers whether the offline-best architecture is still the best choice once FPGA constraints are applied.

Framed this way, the offline benchmark stops being "a number to compare a single implementation against" and becomes "the top of a trade-off curve being mapped out" — a stronger and more complete use of the reference standard than treating it as a target to simply replicate on hardware.

---

## Key Concepts at a Glance

| Concept | What it means |
|---------|---------------|
| **FPGA** | Reconfigurable chip — logic is defined by a bitstream, not fixed in silicon |
| **Bitstream** | Binary file that programs every switch and connection inside the FPGA |
| **LUT** | Look-Up Table — the basic logic building block of an FPGA |
| **DSP Block** | Dedicated multiplier/accumulator unit, critical for neural network math |
| **BRAM** | On-chip memory — limited, so weight storage must be carefully managed |
| **HDL** | Hardware Description Language (VHDL, Verilog) — describes circuits |
| **HLS** | High-Level Synthesis — compiles C/C++ or model descriptions into HDL |
| **Quantization** | Reducing weight precision from float to integer |
| **Pruning** | Removing near-zero weights from a trained network |
| **Loop Unrolling** | Expanding loops into parallel hardware operations |
| **Inference** | Running a trained model on new data to produce predictions |
| **Causal processing** | A computation that only uses past samples — required for real-time streaming |
| **Zero-phase filter** | A non-causal filter (e.g. filtfilt) that needs the full signal; offline-only |
| **Circular buffer** | Fixed-size rolling window of samples, typically stored in BRAM, used for causal windowing |
| **IP core** | A packaged, reusable hardware block with a standardized interface — the same circuit as raw HDL, prepared for visual assembly |
| **Block Design / IP Integrator** | Vivado's visual canvas for wiring IP cores together instead of writing connective HDL by hand |
| **AXI-Lite** | Simple register-style read/write protocol, used for occasional control commands between blocks |
| **AXI-Stream** | Handshake protocol (`TVALID`/`TREADY`/`TLAST`) for continuous, one-directional data flow between blocks |
| **FIFO** | First-In-First-Out buffer, implemented in BRAM, used to absorb rate mismatches between blocks |
| **HLS control protocol (`ap_ctrl`)** | Signals (`ap_start`, `ap_done`, `ap_idle`) that a Vitis HLS block uses to report its own processing status |
| **C/RTL Cosimulation** | Verification step that re-runs a testbench against the generated RTL and compares it to the original C++ output |
| **Analysis window (Tₐ)** | How much signal (ms) must be accumulated before one prediction can be made |
| **Estimation time (Tₑ)** | How long the hardware takes to process one window end-to-end and produce a result |
| **Tₑ/Tₐ ratio** | Overlap indicator: <1 allows overlapping windows (better temporal resolution); >1 forces sequential, non-overlapping processing |

---

## Further Reading

- [An Introduction to FPGAs — SparkFun](https://learn.sparkfun.com/tutorials/programmable-logic-introduction)
- [Efficient Deep Learning — MIT HAN Lab](https://efficientdeeplearning.mit.edu/)
- [FPGA vs GPU for Inference — IEEE Spectrum](https://spectrum.ieee.org)
- [Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference — Google](https://arxiv.org/abs/1712.05877)