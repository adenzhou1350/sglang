use super::{SubmittedChoice, completion_event_stream};
use crate::api_server::guard::AbortGuard;
use crate::api_server::openai::test_utils::senders;
use crate::message::response::{ChunkEvent, ChunkExtras, ResponseItem};
use futures::StreamExt;

#[tokio::test(flavor = "current_thread")]
#[ignore = "temporary production-stream performance harness"]
async fn bench_completion_stream_production() {
    let payload =
        std::env::var("SG_COMPLETION_STREAM_PAYLOAD").expect("SG_COMPLETION_STREAM_PAYLOAD");
    let chunks: usize = std::env::var("SG_COMPLETION_STREAM_CHUNKS")
        .expect("SG_COMPLETION_STREAM_CHUNKS")
        .parse()
        .unwrap();
    let (text, want_logprobs, continuous_usage) = match payload.as_str() {
        "plain_ascii" => ("x".repeat(16), false, false),
        "escaped_text" => ("\\\"line\\n文本".repeat(32), false, false),
        "long_text" => ("x".repeat(4096), false, false),
        "top5_text" => ("token text".repeat(4), true, false),
        "continuous_usage" => ("x".repeat(4), false, true),
        _ => panic!("unsupported SG_COMPLETION_STREAM_PAYLOAD={payload}"),
    };

    let (tx, rx) = tokio::sync::mpsc::channel(chunks + 1);
    for step in 0..chunks {
        let extras = want_logprobs.then(|| {
            Box::new(ChunkExtras {
                out_lp_val: vec![-0.125],
                out_lp_idx: vec![1000],
                out_lp_txt: vec!["token".into()],
                out_top_val: vec![-0.125, -0.25, -0.5, -0.75, -1.0],
                out_top_idx: vec![1000, 1001, 1002, 1003, 1004],
                out_top_lens: vec![5],
                out_top_txt: vec![
                    "token".into(),
                    "a".into(),
                    "b".into(),
                    "c".into(),
                    "d".into(),
                ],
                ..Default::default()
            })
        });
        let output = ChunkEvent {
            rid: "r0".into(),
            token_ids: vec![1000],
            finish_reason: (step + 1 == chunks).then(|| {
                serde_json::from_value(serde_json::json!({
                    "type": "stop",
                    "matched": "</s>"
                }))
                .unwrap()
            }),
            prompt_tokens: 128,
            text: text.clone(),
            completion_tokens: 1,
            extras,
        };
        tx.send(if step + 1 == chunks {
            ResponseItem::Done(output)
        } else {
            ResponseItem::Frame(output)
        })
        .await
        .unwrap();
    }
    drop(tx);

    let submitted = SubmittedChoice {
        index: 0,
        prompt_index: 0,
        rid: "r0".into(),
        echo: String::new(),
        rx,
    };
    let started = std::time::Instant::now();
    let stream = completion_event_stream(
        vec![submitted],
        AbortGuard::new_empty(senders()),
        "cmpl-benchmark-id".into(),
        "benchmark-model".into(),
        1,
        false,
        want_logprobs,
        true,
        continuous_usage,
    );
    futures::pin_mut!(stream);
    let frames: Vec<String> = stream.collect().await;
    let elapsed_ns = started.elapsed().as_nanos();

    assert_eq!(frames.len(), chunks + 2);
    assert_eq!(frames.last().unwrap(), "[DONE]");
    let mut bytes = 0usize;
    let mut output_hash = 0xcbf29ce484222325u64;
    for frame in &frames {
        bytes += frame.len();
        for byte in frame.as_bytes() {
            output_hash = (output_hash ^ u64::from(*byte)).wrapping_mul(0x100000001b3);
        }
        output_hash = (output_hash ^ 0xff).wrapping_mul(0x100000001b3);
    }
    println!(
        "COMPLETION_STREAM_BENCH payload={payload} chunks={chunks} ns={elapsed_ns} frames={} bytes={bytes} hash={output_hash:016x}",
        frames.len()
    );
}
