from pathlib import Path
import sys

from rosbags.highlevel import AnyReader


def main():
    if len(sys.argv) < 2:
        print("Usage: python inspect_ros1_bag.py <bag_path>")
        sys.exit(1)

    bag_path = Path(sys.argv[1])
    if not bag_path.exists():
        raise FileNotFoundError(f"Bag not found: {bag_path}")

    with AnyReader([bag_path]) as reader:
        print(f"Bag: {bag_path}")
        print("\nTopics:")
        for conn in reader.connections:
            print(
                f"  topic={conn.topic} | "
                f"msgtype={conn.msgtype} | "
                f"md5={getattr(conn, 'md5sum', 'N/A')}"
            )


if __name__ == "__main__":
    main()