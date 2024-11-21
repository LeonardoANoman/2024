import React, { useState, useEffect } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  Legend,
  ResponsiveContainer,
  CartesianGrid,
} from "recharts";
import { Card, CardHeader, CardTitle, CardContent } from "./ui/card";
import { bookApi } from "../utils/api";
import { BookStats } from "../types/Book";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

export const StatsCharts: React.FC = () => {
  const [stats, setStats] = useState<BookStats | null>(null);

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const fetchedStats = await bookApi.getBookStats();
        setStats(fetchedStats);
      } catch (error) {
        console.error("Failed to fetch stats:", error);
      }
    };

    fetchStats();
  }, []);

  if (!stats) return <div>Loading...</div>;

  // Custom tooltip formatter
  const formatTooltip = (value: number, name: string) => {
    switch (name) {
      case "Books Read":
        return [`${value} books`, name];
      case "Pages Read":
        return [`${value.toLocaleString()} pages`, name];
      default:
        return [value, name];
    }
  };

  // Calculate trend indicators
  const calculateTrend = (data: number[]) => {
    if (data.length < 2) return 0;
    const lastMonth = data[data.length - 1];
    const previousMonth = data[data.length - 2];
    return ((lastMonth - previousMonth) / previousMonth) * 100;
  };

  const booksTrend = calculateTrend(
    stats.booksByMonth.map((m) => m.book_count)
  );
  const pagesTrend = calculateTrend(
    stats.booksByMonth.map((m) => m.pages_read)
  );

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Reading Stats</CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 gap-6">
          {/* Summary Stats */}
          <div className="grid grid-cols-3 gap-4 bg-gray-50 p-4 rounded-lg">
            <div className="text-center">
              <p className="text-2xl font-bold text-blue-600">
                {stats.totalBooks}
              </p>
              <p className="text-sm text-gray-600">Total Books</p>
              <p
                className={`text-xs ${
                  booksTrend > 0 ? "text-green-600" : "text-red-600"
                }`}
              >
                {booksTrend !== 0 &&
                  `${booksTrend > 0 ? "↑" : "↓"} ${Math.abs(booksTrend).toFixed(
                    1
                  )}% vs last month`}
              </p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-green-600">
                {stats.totalPagesRead?.toLocaleString()}
              </p>
              <p className="text-sm text-gray-600">Pages Read</p>
              <p
                className={`text-xs ${
                  pagesTrend > 0 ? "text-green-600" : "text-red-600"
                }`}
              >
                {pagesTrend !== 0 &&
                  `${pagesTrend > 0 ? "↑" : "↓"} ${Math.abs(pagesTrend).toFixed(
                    1
                  )}% vs last month`}
              </p>
            </div>
            <div className="text-center">
              <p className="text-2xl font-bold text-purple-600">
                {stats.averageRating?.toFixed(1)} ⭐
              </p>
              <p className="text-sm text-gray-600">Avg Rating</p>
            </div>
          </div>

          {/* Charts */}
          <Tabs defaultValue="books" className="w-full">
            <TabsList className="grid w-full grid-cols-2">
              <TabsTrigger value="books">Monthly Books</TabsTrigger>
              <TabsTrigger value="pages">Monthly Pages</TabsTrigger>
            </TabsList>

            <TabsContent value="books">
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats.booksByMonth}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="month_label"
                      tick={{ fill: "#666" }}
                      interval={0}
                      angle={-45}
                      textAnchor="end"
                      height={50}
                    />
                    <YAxis tick={{ fill: "#666" }} allowDecimals={false} />
                    <Tooltip
                      formatter={formatTooltip}
                      contentStyle={{
                        backgroundColor: "rgba(255, 255, 255, 0.9)",
                        border: "1px solid #ccc",
                      }}
                      labelFormatter={(label) => `Month: ${label}`}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="book_count"
                      stroke="#8884d8"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      activeDot={{ r: 8 }}
                      name="Books Read"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>

            <TabsContent value="pages">
              <div className="h-[400px] w-full">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={stats.booksByMonth}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis
                      dataKey="month_label"
                      tick={{ fill: "#666" }}
                      interval={0}
                      angle={-45}
                      textAnchor="end"
                      height={50}
                    />
                    <YAxis tick={{ fill: "#666" }} allowDecimals={false} />
                    <Tooltip
                      formatter={formatTooltip}
                      contentStyle={{
                        backgroundColor: "rgba(255, 255, 255, 0.9)",
                        border: "1px solid #ccc",
                      }}
                      labelFormatter={(label) => `Month: ${label}`}
                    />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="pages_read"
                      stroke="#82ca9d"
                      strokeWidth={2}
                      dot={{ r: 4 }}
                      activeDot={{ r: 8 }}
                      name="Pages Read"
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </CardContent>
    </Card>
  );
};
