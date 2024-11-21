import express from "express";
import {
  addBook,
  getAllBooks,
  getBookStats,
  searchBookImage,
} from "../controllers/bookController.js";

const router = express.Router();

router.post("/", addBook);
router.get("/", getAllBooks);
router.get("/stats", getBookStats);
router.get("/search-image", searchBookImage);

export default router;
