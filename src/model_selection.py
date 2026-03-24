from training_pipeline import run_training_pipeline


def main():
    result = run_training_pipeline(save_artifacts=False)
    report = result["report"]

    print("Model comparison summary")
    print("------------------------")
    for index, row in enumerate(report["model_ranking"], start=1):
        print(
            f"{index}. {row['model']} | "
            f"validation_macro_f1={row['validation_macro_f1']:.4f} | "
            f"validation_accuracy={row['validation_accuracy']:.4f}"
        )

    print(f"\nRecommended model: {report['selected_model']}")
    print(
        "Held-out test metrics: "
        f"accuracy={report['test_metrics']['accuracy']:.4f}, "
        f"macro_f1={report['test_metrics']['macro_f1']:.4f}"
    )


if __name__ == "__main__":
    main()
